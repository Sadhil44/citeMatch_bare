import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.inverted_index import InvertedIndex
from src.retrieval.search import SearchEngine
from src.retrieval.tfidf import TFIDF
from src.retrieval.vectorizer import Vectorizer

docs = json.load(open(REPO_ROOT / "data" / "toy_corpus.json"))

idx = InvertedIndex()
idx.build(docs)

vectorizer = Vectorizer(TFIDF(idx))
engine = SearchEngine(docs, vectorizer)

print(f"Loaded {len(docs)} documents. Try: engine.search('urban heat')")
for r in engine.search("urban heat", k=3):
    print(f"  {r.score:.4f}  doc {r.document_id}  {r.text}")
