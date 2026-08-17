from src.retrieval.bm25 import BM25
from src.retrieval.inverted_index import InvertedIndex
from src.retrieval.search import BM25SearchEngine, SearchEngine
from src.retrieval.tfidf import TFIDF
from src.retrieval.vectorizer import Vectorizer


def precision_at_k(retrieved_docs, relevant_docs, k):
    if k == 0:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]

    relevant_set = set(relevant_docs)

    relevant_retrieved = 0

    for doc_id in retrieved_at_k:
        if doc_id in relevant_set:
            relevant_retrieved += 1

    return relevant_retrieved / k

def recall_at_k(retrieved_docs, relevant_docs, k):
    if not relevant_docs:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]

    relevant_set = set(relevant_docs)

    relevant_retrieved = 0

    for doc_id in retrieved_at_k:
        if doc_id in relevant_set:
            relevant_retrieved += 1

    return relevant_retrieved / len(relevant_set)

def reciprocal_rank(retrieved_docs, relevant_docs):
    relevant_set = set(relevant_docs)

    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_set:
            return 1 / rank

    return 0.0

def build_search_engine(documents, method):
    if method == "bm25":
        return BM25SearchEngine(documents, BM25.from_corpus(documents))

    if method == "tfidf":
        index = InvertedIndex()
        index.build(documents)
        return SearchEngine(documents, Vectorizer(TFIDF(index)))

    raise ValueError(f"Unknown method: {method}")

def evaluate_retriever(evaluation_queries, documents, method, k=5, reranker=None, candidate_k=20):
    engine = build_search_engine(documents, method)
    documents_by_id = {document["id"]: document["text"] for document in documents}

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for example in evaluation_queries:
        query = example["query"]
        relevant_docs = example["relevant_docs"]

        if reranker is None:
            results = engine.search(query, k=k)
            retrieved_docs = [result.document_id for result in results]
        else:
            candidates = engine.search(query, k=candidate_k)
            candidate_pairs = [(result.document_id, result.score) for result in candidates]
            reranked = reranker.rerank(query, candidate_pairs, documents_by_id, k=k)
            retrieved_docs = [doc_id for doc_id, _ in reranked]

        precision_scores.append(precision_at_k(retrieved_docs, relevant_docs, k))
        recall_scores.append(recall_at_k(retrieved_docs, relevant_docs, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved_docs, relevant_docs))

    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    return {
        f"precision@{k}": average_precision,
        f"recall@{k}": average_recall,
        "mrr": mrr
    }
