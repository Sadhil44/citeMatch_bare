from sentence_transformers import SentenceTransformer
import numpy as np

class DenseRetriever:
    def __init__(
        self,
        documents,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_ids =  list(documents.keys())
        document_texts = [documents[doc_id] for doc_id in self.doc_ids]
        self.document_embeddings = self.model.encode(document_texts, normalize_embeddings=True)

    @classmethod
    def from_corpus(cls, corpus, **kwargs):
        documents = {document["id"]: document["text"] for document in corpus}
        return cls(documents, **kwargs)

    def search(self, query, k = 5):
        query_embedding = self.model.encode(query, normalize_embeddings = True)
        scores = np.dot(self.document_embeddings, query_embedding)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_indices = ranked_indices[:k]
        results = []
        for index in ranked_indices:
            doc_id = self.doc_ids[index]
            score = float(scores[index])
            results.append((doc_id, score))
        return results
        
    