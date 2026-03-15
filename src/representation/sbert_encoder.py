"""
Sentence-BERT encoding and persona vector aggregation per time window.
"""



from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


import torch

def load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return SentenceTransformer(model_name, device=device)


def encode_windows(
    windows: Dict[str, List[str]],
    window_ids: List[str],
    model: SentenceTransformer,
    max_posts_per_window: int = 3000,
    batch_size: int = 64,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Encode texts per window with subsampling + batching.
    """
    rng = np.random.RandomState(42)
    persona_vectors = {}
    window_embeddings = {}

    for wid in tqdm(window_ids, desc="Encoding windows"):
        texts = windows[wid]

        if len(texts) > max_posts_per_window:
            idx = rng.choice(len(texts), size=max_posts_per_window, replace=False)
            texts = [texts[i] for i in idx]

        print(f"{wid}: encoding {len(texts)} posts")

        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        window_embeddings[wid] = emb
        mean_vec = np.mean(emb, axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        persona_vectors[wid] = mean_vec

    return window_embeddings, persona_vectors


def get_persona_vectors(
    windows: Dict[str, List[str]],
    window_ids: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_posts_per_window: int = 3000,
    batch_size: int = 64,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], SentenceTransformer]:
    model = load_sbert(model_name)
    window_embeddings, persona_vectors = encode_windows(
        windows,
        window_ids,
        model,
        max_posts_per_window=max_posts_per_window,
        batch_size=batch_size,
    )
    return window_embeddings, persona_vectors, model


