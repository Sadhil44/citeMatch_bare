import requests


BASE_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""

    words_by_position = {}

    for word, positions in inverted_index.items():
        for position in positions:
            words_by_position[position] = word

    ordered_words = [
        words_by_position[position]
        for position in sorted(words_by_position)
    ]

    return " ".join(ordered_words)

def normalize_work(work):
    openalex_id = work.get("id", "")

    if openalex_id:
        paper_id = openalex_id.split("/")[-1]
    else:
        paper_id = ""

    abstract = reconstruct_abstract(
        work.get("abstract_inverted_index")
    )

    return {
        "id": paper_id,
        "title": work.get("title", ""),
        "abstract": abstract,
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "url": openalex_id,
    }

def search_works(
    query,
    per_page=25,
    api_key=None
):
    params = {
        "search": query,
        "per_page": per_page,
    }

    if api_key:
        params["api_key"] = api_key

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    works = data.get("results", [])

    return [
        normalize_work(work)
        for work in works
    ]


