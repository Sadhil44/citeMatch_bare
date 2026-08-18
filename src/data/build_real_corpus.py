import json
from pathlib import Path

from src.data.openalex_client import search_works


DEFAULT_QUERIES = [
    "urban heat island vegetation temperature",
    "machine learning scientific evidence retrieval",
    "climate change surface temperature",
]


def deduplicate_papers(papers):
    unique = {}

    for paper in papers:
        paper_id = paper.get("id")

        if not paper_id:
            continue

        unique[paper_id] = paper

    return list(unique.values())


def build_real_corpus(
    queries=None,
    per_query=50,
    output_path="src/data/real_papers.json",
    api_key=None,
):
    if queries is None:
        queries = DEFAULT_QUERIES

    all_papers = []

    for query in queries:
        print(f"Fetching: {query}")

        papers = search_works(
            query=query,
            per_page=per_query,
            api_key=api_key,
        )

        all_papers.extend(papers)

    unique_papers = deduplicate_papers(
        all_papers
    )

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            unique_papers,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Saved {len(unique_papers)} unique papers "
        f"to {output_file}"
    )

    return unique_papers


if __name__ == "__main__":
    build_real_corpus()