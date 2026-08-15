import math
from collections import Counter
from dataclasses import dataclass

from src.tokenizer import tokenize


@dataclass
class BM25Result:
    document_id: int
    score: float


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.doc_lengths = {}
        self.term_frequencies = {}

        for doc_id, tokens in documents.items():
            self.doc_lengths[doc_id] = len(tokens)
            self.term_frequencies[doc_id] = Counter(tokens)

        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        self.document_frequency = Counter()

        for tokens in self.documents.values():
            unique_items = set(tokens)

            for term in unique_items:
                self.document_frequency[term] += 1

    @classmethod
    def from_corpus(cls, corpus, **kwargs):
        documents = {document["id"]: tokenize(document["text"]) for document in corpus}
        return cls(documents, **kwargs)

    def _idf(self, term):
        N = len(self.documents)
        df = self.document_frequency.get(term, 0)

        return math.log(
            1 + ((N - df + 0.5) / (df + 0.5))
        )

    def score(self, query_tokens, doc_id):
        doc_length = self.doc_lengths[doc_id]
        term_frequencies = self.term_frequencies[doc_id]

        score = 0.0

        for term in query_tokens:
            tf = term_frequencies.get(term, 0)

            if tf == 0:
                continue

            idf = self._idf(term)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b
                + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query, k=5):
        query_tokens = tokenize(query)

        results = [
            BM25Result(document_id=doc_id, score=self.score(query_tokens, doc_id))
            for doc_id in self.documents
        ]

        results.sort(key=lambda result: result.score, reverse=True)

        return results[:k]
