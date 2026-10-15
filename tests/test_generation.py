import json

from src.generation.answer_generator import AnswerGenerator, build_prompt, validate
from src.rag import CitationRAG

PAPERS = [
    {
        "id": "W1",
        "title": "Trees and urban heat",
        "text": "Tree canopy cover lowered daytime land surface temperature by 2 C. Effects were largest in summer.",
    },
    {
        "id": "W2",
        "title": "Impervious surfaces",
        "text": "Asphalt and concrete increase heat retention in dense neighborhoods.",
    },
]

CLAIM = "Urban tree canopy reduces land surface temperature."


def fake_llm(payload):
    def call(system, prompt):
        return "Here you go:\n" + json.dumps(payload)

    return call


def test_prompt_contains_every_paper_id_and_claim():
    prompt = build_prompt(CLAIM, PAPERS)
    assert CLAIM in prompt
    assert 'id="W1"' in prompt and 'id="W2"' in prompt


def test_valid_citation_is_kept():
    parsed = {
        "verdict": "supports",
        "answer": "Canopy cover lowered temperature.",
        "citations": [{"paper_id": "W1", "quote": "Tree canopy cover lowered daytime land surface temperature by 2 C."}],
    }
    result = validate(CLAIM, parsed, PAPERS)
    assert result.verdict == "supports"
    assert [c.paper_id for c in result.citations] == ["W1"]
    assert result.rejected_citations == []


def test_quote_matching_ignores_case_and_whitespace():
    parsed = {
        "verdict": "supports",
        "answer": "x",
        "citations": [{"paper_id": "W1", "quote": "tree canopy   cover LOWERED daytime"}],
    }
    assert len(validate(CLAIM, parsed, PAPERS).citations) == 1


def test_citation_to_unretrieved_paper_is_rejected():
    parsed = {
        "verdict": "supports",
        "answer": "x",
        "citations": [{"paper_id": "W999", "quote": "Tree canopy cover lowered"}],
    }
    result = validate(CLAIM, parsed, PAPERS)
    assert result.citations == []
    assert result.rejected_citations[0]["reason"] == "not retrieved"


def test_invented_quote_is_rejected():
    parsed = {
        "verdict": "supports",
        "answer": "x",
        "citations": [{"paper_id": "W1", "quote": "Canopy cuts temperature by 10 C everywhere."}],
    }
    result = validate(CLAIM, parsed, PAPERS)
    assert result.rejected_citations[0]["reason"] == "quote not in paper"


def test_verdict_downgraded_when_no_citation_survives():
    parsed = {
        "verdict": "supports",
        "answer": "Definitely true.",
        "citations": [{"paper_id": "W999", "quote": "made up"}],
    }
    result = validate(CLAIM, parsed, PAPERS)
    assert result.verdict == "insufficient_evidence"


def test_unknown_verdict_becomes_insufficient_evidence():
    parsed = {"verdict": "probably", "answer": "x", "citations": []}
    assert validate(CLAIM, parsed, PAPERS).verdict == "insufficient_evidence"


def test_generator_handles_unparseable_reply():
    generator = AnswerGenerator(llm=lambda system, prompt: "I think it's true.")
    assert generator.generate(CLAIM, PAPERS).verdict == "insufficient_evidence"


def test_generator_with_no_papers_skips_llm():
    def explode(system, prompt):
        raise AssertionError("LLM should not be called")

    result = AnswerGenerator(llm=explode).generate(CLAIM, [])
    assert result.verdict == "insufficient_evidence"


class FakeRetriever:
    def search(self, query, retrieve_k=20, top_k=5):
        return [("W1", 3.2), ("W2", 0.4)][:top_k]


def test_end_to_end_pipeline_with_fake_retriever_and_llm():
    documents = {p["id"]: p["text"] for p in PAPERS}
    metadata = {p["id"]: {"title": p["title"]} for p in PAPERS}
    generator = AnswerGenerator(
        llm=fake_llm(
            {
                "verdict": "supports",
                "answer": "Paper W1 reports lower temperatures under canopy.",
                "citations": [{"paper_id": "W1", "quote": "Effects were largest in summer."}],
            }
        )
    )
    rag = CitationRAG(FakeRetriever(), documents, metadata, generator)

    result, papers = rag.ask(CLAIM, top_k=2)

    assert [p["id"] for p in papers] == ["W1", "W2"]
    assert result.verdict == "supports"
    assert result.citations[0].paper_id == "W1"
