import json


def load_papers(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def build_paper_text(paper):
    title = paper.get(
        "title",
        ""
    )

    abstract = paper.get(
        "abstract",
        ""
    )

    return f"{title}. {abstract}".strip()

def build_corpus(papers):
    corpus = []
    metadata = {}

    for paper in papers:
        paper_id = paper.get("id")
        if not paper_id:
            continue
        text = build_paper_text(paper)
        if not text:
            continue

        corpus.append({"id": paper_id, "text": text})

        metadata[paper_id] = {
            "title": paper.get(
                "title",
                ""
            ),

            "year": paper.get(
                "year"
            ),

            "doi": paper.get(
                "doi"
            ),

            "url": paper.get(
                "url"
            )
        }
    return (corpus, metadata)