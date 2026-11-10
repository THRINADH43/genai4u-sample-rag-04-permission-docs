"""Hybrid search: BM25 keywords + TF-IDF cosine vectors, fused with reciprocal rank fusion.
It runs only over the chunks the user may see (filtered BEFORE ranking)."""

import math
import re
from collections import Counter, defaultdict

STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "to", "and", "or", "for", "what", "how",
        "who", "which", "do", "does", "my", "we", "our", "i"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def bm25_rank(chunks, query, k1=1.5, b=0.75):
    docs = [Counter(tokenize(c.text)) for c in chunks]
    if not docs:
        return []
    lens = [sum(d.values()) for d in docs]
    avg = sum(lens) / len(lens)
    df = Counter(t for d in docs for t in d)
    scores = defaultdict(float)
    for t in tokenize(query):
        if t not in df:
            continue
        idf = math.log((len(docs) - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for i, d in enumerate(docs):
            if t in d:
                scores[i] += idf * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * lens[i] / avg))
    return [(i, s) for i, s in sorted(scores.items(), key=lambda kv: -kv[1])]


def vector_rank(chunks, query):
    """TF-IDF vectors with cosine similarity. (A sentence-embedding model could replace this;
    I kept TF-IDF so the service and CI run with no downloads.)"""
    docs = [Counter(tokenize(c.text)) for c in chunks]
    if not docs:
        return []
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log(len(docs) / n) + 1 for t, n in df.items()}

    def vec(counts):
        v = {t: n * idf.get(t, 0) for t, n in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    q = vec(Counter(tokenize(query)))
    sims = [(i, sum(q.get(t, 0) * x for t, x in vec(d).items())) for i, d in enumerate(docs)]
    return [(i, s) for i, s in sorted(sims, key=lambda kv: -kv[1]) if s > 0]


def hybrid_search(store, groups, query, k=3):
    chunks = store.visible_chunks(groups)  # permission filter first
    fused = defaultdict(float)
    best_keyword = {}
    for ranking in (bm25_rank(chunks, query), vector_rank(chunks, query)):
        for rank, (i, score) in enumerate(ranking, start=1):
            fused[i] += 1 / (60 + rank)
    for i, s in bm25_rank(chunks, query):
        best_keyword[i] = s
    order = sorted(fused, key=lambda i: -fused[i])[:k]
    return [(chunks[i], best_keyword.get(i, 0.0)) for i in order]
