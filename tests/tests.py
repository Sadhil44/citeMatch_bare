import json
import math

from src.retrieval.inverted_index import InvertedIndex
from src.retrieval.tfidf import TFIDF
from src.retrieval.vectorizer import Vectorizer
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


def test_document_frequency():
    index = _build_toy_index()
    assert index.document_frequency("urban") == 4
    assert index.document_frequency("heat") == 2
    assert index.document_frequency("nonexistent") == 0


def _build_toy_tfidf():
    return TFIDF(_build_toy_index())


def test_idf_of_common_term():
    tfidf = _build_toy_tfidf()
    assert math.isclose(tfidf.idf("urban"), math.log(8 / 4))


def test_idf_of_rare_term():
    tfidf = _build_toy_tfidf()
    assert math.isclose(tfidf.idf("heat"), math.log(8 / 2))


def test_idf_of_unseen_term_is_zero():
    tfidf = _build_toy_tfidf()
    assert tfidf.idf("nonexistent") == 0.0


def test_score_combines_tf_and_idf():
    tfidf = _build_toy_tfidf()
    assert math.isclose(tfidf.score("heat", 2), 2 * math.log(8 / 2))


def test_score_is_zero_when_term_absent_from_document():
    tfidf = _build_toy_tfidf()
    assert tfidf.score("urban", 3) == 0.0


def test_score_is_zero_for_unseen_term():
    tfidf = _build_toy_tfidf()
    assert tfidf.score("nonexistent", 1) == 0.0


def _build_toy_vectorizer():
    return Vectorizer(_build_toy_tfidf())


def test_vocabulary_is_sorted_and_covers_all_terms():
    vectorizer = _build_toy_vectorizer()
    assert vectorizer.vocabulary == sorted(vectorizer.vocabulary)
    assert set(vectorizer.vocabulary) == set(vectorizer.tfidf.index.index.keys())


def test_vectorize_length_matches_vocabulary():
    vectorizer = _build_toy_vectorizer()
    assert len(vectorizer.vectorize(1)) == len(vectorizer.vocabulary)


def test_vectorize_matches_tfidf_score_at_term_position():
    vectorizer = _build_toy_vectorizer()
    vector = vectorizer.vectorize(1)
    urban_index = vectorizer.vocabulary.index("urban")
    assert vector[urban_index] == vectorizer.tfidf.score("urban", 1)


def test_vectorize_is_zero_for_terms_absent_from_document():
    vectorizer = _build_toy_vectorizer()
    vector = vectorizer.vectorize(3)
    heat_index = vectorizer.vocabulary.index("heat")
    assert vector[heat_index] == 0.0


def test_vectorize_all_covers_every_document():
    vectorizer = _build_toy_vectorizer()
    vectors = vectorizer.vectorize_all()
    assert set(vectors.keys()) == vectorizer.tfidf.index.doc_ids
    assert all(len(v) == len(vectorizer.vocabulary) for v in vectors.values())
