class Vectorizer:

    def __init__(self, tfidf):
        self.tfidf = tfidf
        self.vocabulary = sorted(tfidf.index.index.keys()) #build vocab

    def vectorize(self, doc_id):
        return [self.tfidf.score(term, doc_id) for term in self.vocabulary]

    def vectorize_all(self):
        return {
            doc_id: self.vectorize(doc_id)
            for doc_id in sorted(self.tfidf.index.doc_ids)
        }
