"""End-to-end RAG: hybrid retrieval -> cross-encoder rerank -> grounded generation.

Run:
    python -m src.rag "Urban tree canopy reduces land surface temperature"

Needs src/data/real_papers.json (python -m src.data.build_real_corpus)
and ANTHROPIC_API_KEY in the environment.
"""

import sys

from src.data.paper_corpus import build_corpus, load_papers
from src.generation.answer_generator import AnswerGenerator


class CitationRAG:
    def __init__(self, retriever, documents, metadata, generator):
        self.retriever = retriever      # anything with search(query, retrieve_k, top_k) -> [(id, score)]
        self.documents = documents      # id -> paper text
        self.metadata = metadata        # id -> {"title", "year", "doi", "url"}
        self.generator = generator

    @classmethod
    def from_corpus_file(cls, path="src/data/real_papers.json", generator=None):
        from src.retrieval.reranked_search import RerankedSearch

        corpus, metadata = build_corpus(load_papers(path))
        documents = {doc["id"]: doc["text"] for doc in corpus}
        retriever = RerankedSearch.from_corpus(corpus)
        return cls(retriever, documents, metadata, generator or AnswerGenerator())

    def retrieve(self, claim, retrieve_k=20, top_k=5):
        results = self.retriever.search(claim, retrieve_k=retrieve_k, top_k=top_k)
        return [
            {
                "id": doc_id,
                "score": score,
                "title": self.metadata[doc_id].get("title") or "",
                "text": self.documents[doc_id],
            }
            for doc_id, score in results
        ]

    def ask(self, claim, retrieve_k=20, top_k=5):
        papers = self.retrieve(claim, retrieve_k=retrieve_k, top_k=top_k)
        return self.generator.generate(claim, papers), papers


def main():
    claim = " ".join(sys.argv[1:]) or "Urban vegetation reduces land surface temperature."
    rag = CitationRAG.from_corpus_file()
    result, papers = rag.ask(claim)

    print(f"CLAIM:   {result.claim}")
    print(f"VERDICT: {result.verdict}")
    print(f"ANSWER:  {result.answer}\n")
    for citation in result.citations:
        title = rag.metadata[citation.paper_id].get("title")
        print(f"[{citation.paper_id}] {title}")
        print(f'    "{citation.quote}"')
    if result.rejected_citations:
        print(f"\n({len(result.rejected_citations)} citation(s) rejected by the grounding check)")


if __name__ == "__main__":
    main()
