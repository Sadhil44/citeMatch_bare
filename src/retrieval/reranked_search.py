from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.tokenizer import tokenize


class RerankedSearch:
    def __init__(self, tokenized_documents, raw_documents, alpha=0.5, reranker=None):
        self.hybrid = HybridRetriever(tokenized_documents, raw_documents, alpha=alpha)
        self.raw_documents = raw_documents
        self.reranker = reranker or CrossEncoderReranker()

    @classmethod
    def from_corpus(cls, corpus, alpha=0.5, reranker=None):
        tokenized_documents = {
            document["id"]: tokenize(document["text"]) for document in corpus
        }
        raw_documents = {document["id"]: document["text"] for document in corpus}
        return cls(tokenized_documents, raw_documents, alpha=alpha, reranker=reranker)

    def search(self, query, retrieve_k=20, top_k=5):
        candidates = self.hybrid.search(query, k=retrieve_k)
        return self.reranker.rerank(query, candidates, self.raw_documents, k=top_k)
