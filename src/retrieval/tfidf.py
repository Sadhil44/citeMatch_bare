import math

class TFIDF:

    def __init__(self, index):
        self.index = index

    def idf(self, term):
        n = len(self.index.doc_ids)
        df = self.index.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(n / df)

    def score(self, term, doc_id):
        tf = self.index.get_documents(term).get(doc_id, 0)
        return tf * self.idf(term)
