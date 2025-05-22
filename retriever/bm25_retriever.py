from rank_bm25 import BM25Okapi

def bm25_retrieve(query, documents, top_k=50):
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [documents[i] for i, _ in ranked[:top_k]]
