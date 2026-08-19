import json

from src.retrieval.real_search import (
    load_retriever,
    search_real_papers,
)


EVALUATION_PATH = (
    "src/data/real_evaluation_queries.json"
)


def load_queries():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_queries(queries):
    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            queries,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_selection(selected, results):
    if not selected.strip():
        return []

    indexes = [
        int(value.strip()) - 1
        for value
        in selected.split(",")
    ]

    for index in indexes:
        if index < 0 or index >= len(results):
            raise ValueError(
                f"{index + 1} is out of range (1-{len(results)})"
            )

    return [
        results[index]["id"]
        for index in indexes
    ]


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
            relevant_docs = parse_selection(selected, results)
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
