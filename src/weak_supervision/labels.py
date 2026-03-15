"""
Weak supervision: compute cosine distances between adjacent windows;
label stable (low) and shift (high) pairs using percentiles (train windows only).
"""
from typing import List, Tuple
import numpy as np


def compute_distances(
    persona_vectors: dict,
    window_ids: List[str],
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Cosine distance (1 - cos_sim) between consecutive window persona vectors.
    Returns: distances (array), pairs (list of (vec_t, vec_t1, id_t, id_t1))
    """
    distances = []
    pairs = []
    for i in range(len(window_ids) - 1):
        a = persona_vectors[window_ids[i]]
        b = persona_vectors[window_ids[i + 1]]
        d = 1.0 - np.dot(a, b)
        distances.append(d)
        pairs.append((a, b, window_ids[i], window_ids[i + 1]))
    return np.array(distances), pairs


def get_weak_labels(
    distances: np.ndarray,
    pairs: List[Tuple],
    train_window_ids: List[str],
    stable_percentile: float = 30,
    shift_percentile: float = 70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple]]:
    """
    Label pairs as stable (0), shift (1), or unlabeled (-1).
    Only pairs where both windows are in train_window_ids are labeled; rest get -1.

    Returns: X_t, X_t1, y, raw_distances, pair_ids
      - raw_distances: the original SBERT cosine distances for labeled pairs
        (used by the ranking-preservation loss to prevent over-amplification)
    """
    train_set = set(train_window_ids)
    train_indices = [
        i
        for i in range(len(pairs))
        if pairs[i][2] in train_set and pairs[i][3] in train_set
    ]
    if len(train_indices) == 0:
        dist_train = distances
    else:
        dist_train = distances[train_indices]
    stable_thresh = np.percentile(dist_train, stable_percentile)
    shift_thresh = np.percentile(dist_train, shift_percentile)
    labels = []
    for i in range(len(pairs)):
        if i not in train_indices:
            labels.append(-1)
        elif distances[i] <= stable_thresh:
            labels.append(0)
        elif distances[i] >= shift_thresh:
            labels.append(1)
        else:
            labels.append(-1)
    labeled = [
        (pairs[i][0], pairs[i][1], labels[i], distances[i], pairs[i][2], pairs[i][3])
        for i in range(len(pairs))
        if labels[i] != -1
    ]
    if not labeled:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    X_t = np.array([x[0] for x in labeled])
    X_t1 = np.array([x[1] for x in labeled])
    y = np.array([x[2] for x in labeled])
    raw_dist = np.array([x[3] for x in labeled])
    pair_ids = [(x[4], x[5]) for x in labeled]
    return X_t, X_t1, y, raw_dist, pair_ids
