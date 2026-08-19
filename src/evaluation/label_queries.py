from src.evaluation.query_store import (
    load_queries,
    parse_selection,
    save_queries,
)
from src.retrieval.real_search import (
    load_retriever,
    search_real_papers,
)


def main():
    retriever, metadata = (
        load_retriever()
    )

    queries = load_queries()

    for example in queries:
        query = example["query"]

        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = search_real_papers(
            query=query,
            retriever=retriever,
            metadata=metadata,
            retrieve_k=30,
            top_k=10,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"{rank}. "
                f"{result['title']}"
            )

            print(
                f"ID: {result['id']}"
            )

            print(
                f"Year: {result['year']}"
            )

            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

        print()
        selected = input(
            "Relevant result numbers "
            "(example: 1,3,5), "
            "or ENTER for none: "
        )

        try:
            relevant_docs = parse_selection(
                selected,
                [result["id"] for result in results],
            )
        except ValueError as error:
            print(f"Skipping label: {error}")
            continue

        if selected.strip():
            example[
                "relevant_docs"
            ] = relevant_docs

        save_queries(queries)


if __name__ == "__main__":
    main()
