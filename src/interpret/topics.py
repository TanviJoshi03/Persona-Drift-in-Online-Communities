"""
LDA topic extraction for a set of texts (e.g. one time window).
"""
from typing import List, Tuple
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def get_lda_topics(
    texts: List[str],
    n_topics: int = 3,
    top_n_words: int = 8,
) -> List[List[Tuple[str, float]]]:
    """
    Tokenize (content words, len>3), train LDA, return top words per topic.
    Returns list of topics, each topic = list of (word, prob).
    """
    def tokenize(t: str):
        words = [w.lower() for w in t.split() if w.isalpha() and len(w) > 3 and w not in ENGLISH_STOP_WORDS]
        return [w for w in words if w not in {"just", "like", "know", "think", "would", "could", "really", "people", "something"}]
    tokenized = [tokenize(t) for t in texts if len(t.strip()) > 10]
    tokenized = [t for t in tokenized if len(t) > 5]
    if len(tokenized) < 10:
        return [[("insufficient_data", 0.0)]]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=500)
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    if len(corpus) < 5:
        return [[("insufficient_data", 0.0)]]
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        iterations=100,
        alpha="auto",
        random_state=42,
    )
    topics = []
    for k in range(n_topics):
        topics.append(lda.show_topic(k, topn=top_n_words))
    return topics
