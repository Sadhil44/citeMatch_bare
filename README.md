# citeMatch

A from-scratch information retrieval toolkit: tokenization, an inverted index with boolean search, TF-IDF, BM25, cosine similarity, dense/semantic retrieval, and a small evaluation harness (precision@k, recall@k, MRR) for comparing retrievers against each other.

## Setup

```
pip install -r requirements.txt
```

## Running tests

```
python -m pytest -v
```

## Project structure

```
src/
  tokenizer.py               lowercase, strip punctuation, split, drop empty tokens
  retrieval/
    inverted_index.py        InvertedIndex: term -> {doc_id: term_frequency}, boolean AND/OR/NOT search
    tfidf.py                 TFIDF: idf(term), score(term, doc_id) on top of an InvertedIndex
    vectorizer.py             Vectorizer: dense per-document tf-idf vectors over a fixed vocabulary
    similarity.py             dot_product, magnitude, cosine_similarity
    bm25.py                   BM25: Okapi BM25 scoring and search
    dense_retrieval.py        DenseRetriever: sentence-transformer embeddings + cosine search
    hybrid_retriever.py       HybridRetriever: min-max-normalized blend of BM25 + DenseRetriever
    cross_encoder_reranker.py CrossEncoderReranker: reranks a candidate list with a cross-encoder model
    reranked_search.py        RerankedSearch: HybridRetriever candidates reranked by CrossEncoderReranker
    search.py                 SearchEngine (tf-idf + cosine) and BM25SearchEngine, both returning SearchResult
    real_search.py            load_retriever/search_real_papers: RerankedSearch wired to the real corpus
  data/
    toy_corpus.json           8-document toy corpus used throughout the tests
    openalex_client.py        search_works: query the OpenAlex API for papers
    paper_corpus.py           build_corpus: normalize raw papers into {"id", "text"} + metadata
    build_real_corpus.py      fetches papers from OpenAlex and saves them to real_papers.json
    load_real_corpus.py       load_real_corpus(): loads a saved real_papers.json into a corpus
  evaluation/
    evaluation.py             precision_at_k, recall_at_k, reciprocal_rank, evaluate_retriever
    label_queries.py          interactive CLI to hand-label relevant_docs for real queries
tests/
  tests.py                   full test suite
repl.py                      interactive shell preloaded with an indexed toy corpus
```

## Quick start

```
python -i repl.py
```

Drops you into a REPL with `engine`, `idx`, `vectorizer`, and `docs` already loaded:

```python
>>> engine.search("urban heat")
```

Or build any of the retrievers directly:

```python
import json
from src.retrieval.bm25 import BM25
from src.retrieval.dense_retrieval import DenseRetriever

docs = json.load(open("src/data/toy_corpus.json"))

bm25 = BM25.from_corpus(docs)
bm25.search("urban heat")

dense = DenseRetriever.from_corpus(docs)
dense.search("why are cities warmer than the countryside")
```

## Evaluating retrievers

```python
from src.evaluation.evaluation import evaluate_retriever

queries = [{"query": "urban heat", "relevant_docs": [2, 8]}]
evaluate_retriever(queries, docs, method="bm25", k=5)   # or method="tfidf"
```
