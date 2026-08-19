from src.data.paper_corpus import load_papers, build_corpus
from src.evaluation.query_store import (
    load_queries,
    parse_selection,
    save_queries,
)
from src.retrieval.hybrid_retriever import HybridRetriever


CORPUS_PATH = "src/data/real_papers.json"

def build_retrievers():
    papers = load_papers(CORPUS_PATH)
    corpus, metadata = build_corpus(papers)
    hybrid = HybridRetriever.from_corpus(corpus)
    return hybrid.bm25, hybrid.dense, hybrid, metadata

def get_candidate_pool(
    query,
    bm25,
    dense,
    hybrid,
    top_k=10,
):
    bm25_results = [
        (result.document_id, result.score)
        for result in bm25.search(query, k=top_k)
    ]

    dense_results = dense.search(
        query,
        k=top_k,
    )

    hybrid_results = hybrid.search(
        query,
        k=top_k,
    )

    candidate_ids = []

    seen = set()

    for results in [
        bm25_results,
        dense_results,
        hybrid_results,
    ]:
        for doc_id, _ in results:
            if doc_id not in seen:
                candidate_ids.append(
                    doc_id
                )

                seen.add(
                    doc_id
                )

    return candidate_ids

def main():
    bm25, dense, hybrid, metadata = (
        build_retrievers()
    )

    queries = load_queries()

    for example in queries:
        query = example["query"]

        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        candidate_ids = get_candidate_pool(
            query=query,
            bm25=bm25,
            dense=dense,
            hybrid=hybrid,
            top_k=10,
        )

        for index, doc_id in enumerate(
            candidate_ids,
            start=1,
        ):
            paper = metadata[doc_id]

            print()
            print(
                f"{index}. {paper['title']}"
            )

            print(
                f"ID: {doc_id}"
            )

            print(
                f"Year: {paper['year']}"
            )

        print()

        selected = input(
            "Relevant papers "
            "(example: 1,4,7), "
            "or ENTER for none: "
        )

        try:
            relevant_docs = parse_selection(selected, candidate_ids)
        except ValueError as error:
            print(f"Skipping label: {error}")
            continue

        if selected.strip():
            example["relevant_docs"] = (
                relevant_docs
            )

        save_queries(
            queries
        )


if __name__ == "__main__":
    main()
