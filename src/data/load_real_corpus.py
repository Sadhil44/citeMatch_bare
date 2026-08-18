from src.data.paper_corpus import (
    load_papers,
    build_corpus
)


def load_real_corpus(path="src/data/real_papers.json"):
    papers = load_papers(path)
    return build_corpus(papers)


if __name__ == "__main__":
    corpus, metadata = load_real_corpus()
    print(f"Loaded {len(corpus)} papers")
