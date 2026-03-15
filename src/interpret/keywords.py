"""
Keyword and phrase shift analysis between two text collections (e.g. before/after a change point).
"""
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def keyword_shifts(
    texts_before: List[str],
    texts_after: List[str],
    min_delta: float = 0.01,
    top_n: int = 15,
    min_len: int = 4,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    TF-IDF delta between "before" and "after" doc (each doc = joined texts).
    Returns (emerging: [(word, delta)], declining: [(word, -delta)])
    """
    custom_stop = set(ENGLISH_STOP_WORDS) | {"like", "know", "just", "think", "ve", "don", "ll", "really", "want", "got", "thing", "things"}
    before_text = " ".join(texts_before)
    after_text = " ".join(texts_after)
    vec = TfidfVectorizer(
        max_features=1000,
        stop_words=list(custom_stop),
        token_pattern=r"\b[a-zA-Z]{" + str(min_len) + r",}\b",
    )
    X = vec.fit_transform([before_text, after_text]).toarray()
    delta = X[1] - X[0]
    names = vec.get_feature_names_out()
    significant = np.where(np.abs(delta) > min_delta)[0]
    if len(significant) == 0:
        return [], []
    order = np.argsort(np.abs(delta[significant]))[-top_n:][::-1]
    emerging = []
    declining = []
    for idx in order:
        i = significant[idx]
        if delta[i] > 0:
            emerging.append((names[i], float(delta[i])))
        else:
            declining.append((names[i], float(delta[i])))
    return emerging[:top_n], declining[:top_n]


def phrase_shifts(
    texts_before: List[str],
    texts_after: List[str],
    ngram_range: Tuple[int, int] = (2, 3),
    top_n: int = 10,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Bigram/trigram frequency delta (normalized counts) between before and after."""
    vec = CountVectorizer(ngram_range=ngram_range, max_features=500, min_df=2, stop_words="english")
    before_text = " ".join(texts_before)
    after_text = " ".join(texts_after)
    X = vec.fit_transform([before_text, after_text]).toarray()
    before_freq = X[0] / (X[0].sum() + 1e-9)
    after_freq = X[1] / (X[1].sum() + 1e-9)
    delta = after_freq - before_freq
    names = vec.get_feature_names_out()
    top_inc = np.argsort(delta)[-top_n:][::-1]
    top_dec = np.argsort(delta)[:top_n]
    emerging = [(names[i], float(delta[i])) for i in top_inc if delta[i] > 1e-5]
    declining = [(names[i], float(delta[i])) for i in top_dec if delta[i] < -1e-5]
    return emerging, declining
