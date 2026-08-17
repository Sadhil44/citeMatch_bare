from src.retrieval.bm25 import BM25
from src.retrieval.dense_retrieval import DenseRetriever
from src.tokenizer import tokenize

def min_max_normalize(scores):
    if not scores:
        return {}

    values = list(scores.values())

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {
            doc_id: 0.0
            for doc_id in scores
        }

    return {
        doc_id: (score - minimum) / (maximum - minimum)
        for doc_id, score in scores.items()
    }

class HybridRetriever:
    def __init__(self, tokenized_documents, raw_documents, alpha=0.5):
        self.alpha = alpha
        self.bm25 = BM25(tokenized_documents)
        self.dense = DenseRetriever(raw_documents)

    @classmethod
    def from_corpus(cls, corpus, alpha=0.5):
        tokenized_documents = {
            document["id"]: tokenize(document["text"]) for document in corpus
        }
        raw_documents = {document["id"]: document["text"] for document in corpus}
        return cls(tokenized_documents, raw_documents, alpha=alpha)

    def search(self, query, k = 5):
        bm25_results = self.bm25.search(query, k=len(self.bm25.documents))
        dense_results = self.dense.search(query, k = len(self.dense.documents))
        bm25_scores = {result.document_id: result.score for result in bm25_results}
        dense_scores = dict(dense_results)
        bm25_scores = min_max_normalize(bm25_scores)
        dense_scores = min_max_normalize(dense_scores)

        all_doc_ids = set(bm25_scores) | set(dense_scores)

        combined_scores = {}

        for doc_id in all_doc_ids:

            bm25_score = bm25_scores.get(
                doc_id,
                0.0
            )

            dense_score = dense_scores.get(
                doc_id,
                0.0
            )

            combined_score = (
                (1 - self.alpha) * bm25_score
                +
                self.alpha * dense_score
            )

            combined_scores[doc_id] = (
                combined_score
            )

        ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_results[:k]
