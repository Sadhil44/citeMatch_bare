"""Generation step: turn retrieved papers into a grounded, cited answer.

The LLM only sees the papers retrieval returned. Its output is then checked
in code before anyone sees it:

- every cited paper must be one of the retrieved papers, and
- every quoted piece of evidence must appear word-for-word in that paper's text.

Citations that fail either check are dropped. If none survive, the answer
falls back to "insufficient_evidence" instead of returning an unsupported claim.
"""

import json
import os
import re
from dataclasses import dataclass, field


VERDICTS = ("supports", "refutes", "mixed", "insufficient_evidence")

DEFAULT_MODEL = os.environ.get("CITEMATCH_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You assess scientific claims using ONLY the papers provided. "
    "Do not use outside knowledge. If the papers do not address the claim, "
    "answer insufficient_evidence. Respond with JSON only."
)


@dataclass
class Citation:
    paper_id: str
    quote: str


@dataclass
class GeneratedAnswer:
    claim: str
    verdict: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    rejected_citations: list[dict] = field(default_factory=list)


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def build_prompt(claim, papers):
    """papers: list of {"id", "title", "text"} in retrieval order."""
    blocks = []
    for paper in papers:
        blocks.append(
            f'<paper id="{paper["id"]}">\n'
            f'Title: {paper["title"]}\n'
            f'{paper["text"]}\n'
            f"</paper>"
        )

    return (
        f"Claim: {claim}\n\n"
        "Papers:\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON with exactly these keys:\n"
        '{"verdict": one of ' + ", ".join(f'"{v}"' for v in VERDICTS) + ",\n"
        ' "answer": 2-4 sentences explaining the verdict,\n'
        ' "citations": [{"paper_id": an id from above, '
        '"quote": a sentence copied exactly from that paper}]}'
    )


def parse_response(raw_text):
    """Pull the first JSON object out of the model's reply."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("model response contained no JSON object")
    return json.loads(match.group(0))


def validate(claim, parsed, papers):
    """Keep only citations that point at a retrieved paper and quote it exactly."""
    texts = {paper["id"]: _normalize(paper["title"] + ". " + paper["text"]) for paper in papers}

    kept, rejected = [], []
    for citation in parsed.get("citations") or []:
        paper_id = str(citation.get("paper_id", ""))
        quote = str(citation.get("quote", ""))

        if paper_id not in texts:
            rejected.append({"paper_id": paper_id, "quote": quote, "reason": "not retrieved"})
        elif not quote.strip() or _normalize(quote) not in texts[paper_id]:
            rejected.append({"paper_id": paper_id, "quote": quote, "reason": "quote not in paper"})
        else:
            kept.append(Citation(paper_id=paper_id, quote=quote))

    verdict = parsed.get("verdict")
    answer = str(parsed.get("answer", "")).strip()

    if verdict not in VERDICTS:
        verdict = "insufficient_evidence"

    if verdict != "insufficient_evidence" and not kept:
        verdict = "insufficient_evidence"
        answer = "The retrieved papers did not provide verifiable evidence for this claim."

    return GeneratedAnswer(
        claim=claim,
        verdict=verdict,
        answer=answer,
        citations=kept,
        rejected_citations=rejected,
    )


def anthropic_llm(model=DEFAULT_MODEL, max_tokens=800):
    """Default LLM: returns a function(system, prompt) -> reply text."""
    from anthropic import Anthropic

    client = Anthropic()

    def call(system, prompt):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return call


class AnswerGenerator:
    def __init__(self, llm=None):
        # llm: any function(system, prompt) -> str. Tests pass a fake one.
        self.llm = llm or anthropic_llm()

    def generate(self, claim, papers):
        if not papers:
            return GeneratedAnswer(
                claim=claim,
                verdict="insufficient_evidence",
                answer="No papers were retrieved for this claim.",
            )

        raw = self.llm(SYSTEM_PROMPT, build_prompt(claim, papers))

        try:
            parsed = parse_response(raw)
        except (ValueError, json.JSONDecodeError):
            return GeneratedAnswer(
                claim=claim,
                verdict="insufficient_evidence",
                answer="The model returned an unreadable response.",
            )

        return validate(claim, parsed, papers)
