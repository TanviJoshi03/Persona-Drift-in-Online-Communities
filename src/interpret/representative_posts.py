"""
Representative posts: those closest to the window's persona vector in embedding space.
"""
from typing import Dict, List
import numpy as np


def get_representative_posts(
    window_embeddings: np.ndarray,
    persona_vector: np.ndarray,
    texts: List[str],
    top_n: int = 3,
) -> List[str]:
    """Return texts whose embeddings have highest cosine similarity to the persona vector."""
    sims = np.dot(window_embeddings, persona_vector)
    idx = np.argsort(sims)[-top_n:][::-1]
    return [texts[i] for i in idx]
