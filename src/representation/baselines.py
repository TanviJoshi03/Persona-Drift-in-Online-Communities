"""
Baseline drift signals: LDA (Jensen-Shannon) and TF-IDF (JSD) between consecutive windows.
"""
from typing import Dict, List
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim import corpora
from gensim.models import LdaModel


def compute_lda_drift(
    windows: Dict[str, List[str]],
    window_ids: List[str],
    n_topics: int = 10,
) -> List[float]:
    """Drift = Jensen-Shannon divergence between LDA topic distributions of consecutive windows."""
    all_texts = []
    window_docs = []
    for wid in window_ids:
        for t in windows[wid]:
            toks = [
                w.lower()
                for w in t.split()
                if w.isalpha() and len(w) > 2 and w not in ENGLISH_STOP_WORDS
            ]
            if len(toks) > 5:
                all_texts.append(toks)
                window_docs.append(wid)
    if len(all_texts) < 50:
        return [0.0] * max(0, len(window_ids) - 1)
    dictionary = corpora.Dictionary(all_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=500)
    corpus = [dictionary.doc2bow(t) for t in all_texts]
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=5,
        random_state=42,
        alpha="auto",
    )
    win_dist = {wid: np.zeros(n_topics) for wid in window_ids}
    for j, wid in enumerate(window_docs):
        topics = lda.get_document_topics(corpus[j], minimum_probability=0)
        for ti, p in topics:
            win_dist[wid][ti] += p
    for wid in window_ids:
        s = win_dist[wid].sum()
        win_dist[wid] = win_dist[wid] / s if s > 0 else win_dist[wid] + 1.0 / n_topics
    return [
        float(jensenshannon(win_dist[window_ids[i]], win_dist[window_ids[i + 1]]))
        for i in range(len(window_ids) - 1)
    ]


def compute_tfidf_drift(
    windows: Dict[str, List[str]],
    window_ids: List[str],
    max_features: int = 500,
) -> List[float]:
    """Drift = Jensen-Shannon divergence between normalized TF-IDF vectors of consecutive windows."""
    all_texts = [" ".join(windows[wid]) for wid in window_ids]
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )
    X = vec.fit_transform(all_texts).toarray()
    X = (X + 1e-9) / (X.sum(axis=1, keepdims=True) + 1e-9)
    return [float(jensenshannon(X[i], X[i + 1])) for i in range(len(window_ids) - 1)]


def compute_sbert_baseline_drift(
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
) -> List[float]:
    """Drift = 1 - cosine_sim between raw SBERT persona vectors (cosine distance)."""
    out = []
    for i in range(len(window_ids) - 1):
        a = persona_vectors[window_ids[i]]
        b = persona_vectors[window_ids[i + 1]]
        out.append(float(1.0 - np.dot(a, b)))
    return out


def compute_mmd_drift(
    window_embeddings: Dict[str, np.ndarray],
    window_ids: List[str],
    sigma: float = None,
    max_samples: int = 300,
) -> List[float]:
    """
    Maximum Mean Discrepancy between full embedding distributions of consecutive windows.
    MMD compares distributions (not just means), capturing higher-order differences
    that cosine distance of means misses.
    Uses RBF kernel; sigma defaults to median pairwise distance (median heuristic).
    """
    def _subsample(X, k):
        if len(X) <= k:
            return X
        idx = np.random.RandomState(42).choice(len(X), k, replace=False)
        return X[idx]

    def _rbf_kernel(X, Y, s):
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_dist = XX + YY.T - 2.0 * X @ Y.T
        return np.exp(-sq_dist / (2.0 * s ** 2))

    def _mmd2(X, Y, s):
        Kxx = _rbf_kernel(X, X, s)
        Kyy = _rbf_kernel(Y, Y, s)
        Kxy = _rbf_kernel(X, Y, s)
        return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    drift = []
    for i in range(len(window_ids) - 1):
        X = _subsample(window_embeddings[window_ids[i]], max_samples)
        Y = _subsample(window_embeddings[window_ids[i + 1]], max_samples)
        if sigma is None:
            combined = np.vstack([X[:50], Y[:50]])
            from scipy.spatial.distance import pdist
            dists = pdist(combined, "euclidean")
            s = float(np.median(dists)) if len(dists) > 0 else 1.0
        else:
            s = sigma
        val = max(0.0, _mmd2(X, Y, s))
        drift.append(float(np.sqrt(val)))
    return drift
