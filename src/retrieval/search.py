from dataclasses import dataclass

from src.retrieval.similarity import cosine_similarity
from src.retrieval.vectorizer import vectorize_query
from src.tokenizer import tokenize


@dataclass
class SearchResult:
    document_id: int
    text: str
    score: float


class SearchEngine:

    def __init__(self, documents, vectorizer):
        self.documents = documents
        self.vectorizer = vectorizer
        self.document_vectors = vectorizer.vectorize_all()

    def search(self, query, k: int = 5):
        query_vector = vectorize_query(query, self.vectorizer)

        results = [
            SearchResult(
                document_id=document["id"],
                text=document["text"],
                score=cosine_similarity(
                    query_vector, self.document_vectors[document["id"]]
                ),
            )
            for document in self.documents
        ]

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:k]


class BM25SearchEngine:

    def __init__(self, documents, bm25):
        self.documents = documents
        self.bm25 = bm25

    def search(self, query, k: int = 5):
        query_tokens = tokenize(query)

        results = [
            SearchResult(
                document_id=document["id"],
                text=document["text"],
                score=self.bm25.score(query_tokens, document["id"]),
            )
            for document in self.documents
        ]

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:k]
