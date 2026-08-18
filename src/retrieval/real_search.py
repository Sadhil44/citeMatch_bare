from src.data.paper_corpus import load_papers, build_corpus

from src.retrieval.reranked_search import RerankedSearch

def load_retriever(
    corpus_path="src/data/real_papers.json"
):
    papers = load_papers(
        corpus_path
    )

    corpus, metadata = build_corpus(papers)

    retriever = RerankedSearch.from_corpus(corpus)

    return retriever, metadata

def search_real_papers(
    query,
    retriever,
    metadata,
    retrieve_k=20,
    top_k=5,
):
    results = retriever.search(
        query=query,
        retrieve_k=retrieve_k,
        top_k=top_k,
    )

    output = []

    for paper_id, score in results:
        paper_metadata = metadata[
            paper_id
        ]

        output.append(
            {
                "id": paper_id,
                "score": score,
                "title": paper_metadata.get(
                    "title"
                ),
                "year": paper_metadata.get(
                    "year"
                ),
                "doi": paper_metadata.get(
                    "doi"
                ),
                "url": paper_metadata.get(
                    "url"
                ),
            }
        )

    return output

if __name__ == "__main__":
    retriever, metadata = (
        load_retriever()
    )
