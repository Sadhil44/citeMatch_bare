from collections import Counter

from src.tokenizer import tokenize

class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.doc_ids = set()

    def add_document(self, document):
        self.doc_ids.add(document["id"])
        term_counts = Counter(tokenize(document["text"]))
        for term, count in term_counts.items():
            postings = self.index.setdefault(term, {})
            postings[document["id"]] = count

    def build(self, documents):
        for document in documents:
            self.add_document(document)

    def get_documents(self, term):
        tokens = tokenize(term)
        if not tokens:
            return {}
        return self.index.get(tokens[0], {})

    def search(self, query):
        operators = {"AND", "OR", "NOT"}
        tokens = query.split()

        result = None
        combine = "AND"
        negate = False

        for token in tokens:
            if token == "NOT":
                negate = True
                continue
            if token in operators:
                combine = token
                continue

            docs = set(self.get_documents(token).keys())
            if negate:
                docs = self.doc_ids - docs
                negate = False

            if result is None:
                result = docs
            elif combine == "AND":
                result &= docs
            elif combine == "OR":
                result |= docs

        return result or set()

    def document_frequency(self, term):
        return len(self.get_documents(term))

    