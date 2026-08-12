import json

from src.retrieval.inverted_index import InvertedIndex
from src.tokenizer import tokenize

with open("data/toy_corpus.json") as f:
    TOY_CORPUS = json.load(f)


def test_lowercase():
    assert tokenize("Urban VEGETATION Reduces") == ["urban", "vegetation", "reduces"]


def test_punctuation():
    assert tokenize("Urban vegetation reduces temperature!") == [
        "urban",
        "vegetation",
        "reduces",
        "temperature",
    ]


def test_whitespace():
    assert tokenize("  urban   vegetation\treduces\n\ntemperature  ") == [
        "urban",
        "vegetation",
        "reduces",
        "temperature",
    ]


def test_empty_tokens_are_dropped():
    assert tokenize("...!!!   ,,,") == []


def test_empty_string():
    assert tokenize("") == []


def _build_toy_index():
    index = InvertedIndex()
    index.build(TOY_CORPUS)
    return index


def test_term_appears_in_expected_documents():
    index = _build_toy_index()
    assert index.get_documents("urban") == {1: 1, 2: 1, 7: 1, 8: 1}


def test_term_frequency_within_a_document():
    index = _build_toy_index()
    assert index.get_documents("heat") == {2: 2, 8: 1}


def test_unknown_term_returns_empty_dict():
    index = _build_toy_index()
    assert index.get_documents("nonexistent") == {}


def test_empty_query_returns_empty_dict():
    index = _build_toy_index()
    assert index.get_documents("") == {}


def test_search_and():
    index = _build_toy_index()
    assert index.search("urban AND heat") == {2, 8}


def test_search_or():
    index = _build_toy_index()
    assert index.search("urban OR heat") == {1, 2, 7, 8}


def test_search_not_after_term():
    index = _build_toy_index()
    assert index.search("urban NOT heat") == {1, 7}


def test_search_leading_not():
    index = _build_toy_index()
    assert index.search("NOT urban") == {3, 4, 5, 6}


def test_search_unknown_term_returns_empty_set():
    index = _build_toy_index()
    assert index.search("nonexistent AND urban") == set()


def test_search_empty_query_returns_empty_set():
    index = _build_toy_index()
    assert index.search("") == set()
