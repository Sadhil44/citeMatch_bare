from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)
    def rerank(
        self,
        query,
        candidates,
        documents, 
        k = 5
    ):
        pairs = []
        candidate_ids = []
        for doc_id, _ in candidates:
            pairs.append(
                [
                    query,
                    documents[doc_id]
                ]
            )

            candidate_ids.append(doc_id)
        scores = self.model.predict(pairs)
        reranked_results = []
        for doc_id, score in zip(
            candidate_ids,
            scores
        ):
            reranked_results.append(
                (doc_id, float(score))
            )

        reranked_results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return reranked_results[:k]
