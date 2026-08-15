import json
import math

from src.retrieval.bm25 import BM25
from src.retrieval.inverted_index import InvertedIndex
from src.retrieval.search import BM25SearchEngine, SearchEngine, SearchResult
from src.retrieval.similarity import cosine_similarity, dot_product, magnitude
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


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0


def test_magnitude():
    assert magnitude([3, 4]) == 5.0


def test_cosine_similarity_of_identical_vectors_is_one():
    vectorizer = _build_toy_vectorizer()
    vector = vectorizer.vectorize(1)
    assert math.isclose(cosine_similarity(vector, vector), 1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_similarity_with_zero_vector_is_zero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_similarity_ranks_related_documents_higher():
    vectorizer = _build_toy_vectorizer()
    vectors = vectorizer.vectorize_all()
    # doc 1 and doc 4 both discuss land surface temperature; doc 2 is about
    # impervious surfaces/heat retention and shares less vocabulary with doc 1.
    related = cosine_similarity(vectors[1], vectors[4])
    unrelated = cosine_similarity(vectors[1], vectors[2])
    assert related > unrelated


def _build_toy_search_engine():
    return SearchEngine(TOY_CORPUS, _build_toy_vectorizer())


def test_search_returns_search_results():
    engine = _build_toy_search_engine()
    results = engine.search("urban heat")
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_respects_k():
    engine = _build_toy_search_engine()
    assert len(engine.search("urban heat", k=3)) == 3
    assert len(engine.search("urban heat", k=2)) == 2


def test_search_default_k_is_five():
    engine = _build_toy_search_engine()
    assert len(engine.search("urban")) == 5


def test_search_results_sorted_descending():
    engine = _build_toy_search_engine()
    results = engine.search("urban heat", k=8)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_ranks_best_matching_document_first():
    engine = _build_toy_search_engine()
    results = engine.search("urban heat", k=1)
    # doc 2 mentions "heat" twice and "urban" once, the strongest match
    # for this query among the toy corpus.
    assert results[0].document_id == 2


def test_search_empty_query_scores_everything_zero():
    engine = _build_toy_search_engine()
    results = engine.search("", k=8)
    assert all(r.score == 0.0 for r in results)


def _build_toy_bm25():
    return BM25.from_corpus(TOY_CORPUS)


def test_bm25_document_frequency_matches_inverted_index():
    bm25 = _build_toy_bm25()
    index = _build_toy_index()
    assert bm25.document_frequency["urban"] == index.document_frequency("urban")
    assert bm25.document_frequency["heat"] == index.document_frequency("heat")


def test_bm25_score_is_zero_when_no_query_terms_present():
    bm25 = _build_toy_bm25()
    assert bm25.score(["urban", "heat"], 3) == 0.0


def test_bm25_score_is_zero_for_unseen_term():
    bm25 = _build_toy_bm25()
    assert bm25.score(["nonexistent"], 1) == 0.0


def test_bm25_score_rewards_higher_term_frequency():
    bm25 = _build_toy_bm25()
    # doc 2 mentions "heat" twice, doc 8 once; both are similar length.
    assert bm25.score(["heat"], 2) > bm25.score(["heat"], 8)


def test_bm25_search_respects_k():
    bm25 = _build_toy_bm25()
    assert len(bm25.search("urban heat", k=3)) == 3
    assert len(bm25.search("urban heat", k=8)) == 8


def test_bm25_search_results_sorted_descending():
    bm25 = _build_toy_bm25()
    results = bm25.search("urban heat", k=8)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_search_ranks_best_matching_document_first():
    bm25 = _build_toy_bm25()
    results = bm25.search("urban heat", k=1)
    assert results[0].document_id == 2


def test_bm25_search_empty_query_scores_everything_zero():
    bm25 = _build_toy_bm25()
    results = bm25.search("", k=8)
    assert all(r.score == 0.0 for r in results)


def _build_toy_bm25_search_engine():
    return BM25SearchEngine(TOY_CORPUS, _build_toy_bm25())


def test_bm25_search_engine_returns_search_results_with_text():
    engine = _build_toy_bm25_search_engine()
    results = engine.search("urban heat")
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.text for r in results)


def test_bm25_search_engine_matches_raw_bm25_score():
    bm25 = _build_toy_bm25()
    engine = BM25SearchEngine(TOY_CORPUS, bm25)
    results = engine.search("urban heat", k=8)
    for result in results:
        assert result.score == bm25.score(tokenize("urban heat"), result.document_id)


def test_bm25_search_engine_respects_k():
    engine = _build_toy_bm25_search_engine()
    assert len(engine.search("urban heat", k=3)) == 3


def test_bm25_search_engine_ranks_best_matching_document_first():
    engine = _build_toy_bm25_search_engine()
    results = engine.search("urban heat", k=1)
    assert results[0].document_id == 2
