"""
Evaluation metrics for persona drift detection.

Drift-curve metrics:
  temporal_smoothness, change_point_rank, spearman_correlation

Ranking / correlation metrics (continuous drift curve vs ground truth):
  kendall_tau, r_squared, ndcg_at_k, ranking_metrics

Change-point detection metrics:
  hit_at_k, detection_delay, mean_reciprocal_rank, cpd_precision_recall_f1
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau


# ── Drift curve metrics ──────────────────────────────────────────────────

def temporal_smoothness(curve: List[float]) -> float:
    """Mean absolute first difference. Lower = smoother."""
    if len(curve) <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(curve))))


def change_point_rank(drift_curve: List[float], true_change_idx: Optional[int] = None) -> dict:
    """Where the top peak ranks among all transitions."""
    arr = np.array(drift_curve)
    top_idx = int(np.argmax(arr))
    rank_order = np.argsort(arr)[::-1]
    peak_ratio = float(arr[top_idx] / (arr.mean() + 1e-9))
    result = {"top_peak_idx": top_idx, "peak_height_ratio": round(peak_ratio, 3)}
    if true_change_idx is not None:
        rank = int(np.where(rank_order == true_change_idx)[0][0]) + 1
        result["true_change_rank"] = rank
        result["true_change_in_top3"] = rank <= 3
    return result


def spearman_correlation(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation between two drift curves."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    r, _ = spearmanr(a, b)
    return float(r) if not np.isnan(r) else 0.0


# ── Ranking / correlation metrics ─────────────────────────────────────────

def kendall_tau(a: List[float], b: List[float]) -> float:
    """Kendall's tau rank correlation between two drift curves."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    tau, _ = kendalltau(a, b)
    return round(float(tau), 4) if not np.isnan(tau) else 0.0


def r_squared(predicted: List[float], ground_truth: List[float]) -> float:
    """
    Coefficient of determination (R²) between predicted drift and ground truth.
    Both curves are min-max normalised to [0, 1] before computing R².
    """
    p = np.array(predicted, dtype=float)
    g = np.array(ground_truth, dtype=float)
    if len(p) != len(g) or len(p) < 2:
        return 0.0
    # normalise
    p = (p - p.min()) / (p.max() - p.min() + 1e-9)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return round(1.0 - ss_res / ss_tot, 4)


def ndcg_at_k(predicted: List[float], ground_truth: List[float], k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain @ k.
    Measures whether the method's top-k highest-drift transitions
    coincide with the actual top-k highest-drift transitions from the
    ground truth.

    Relevance scores are the ground truth drift values.
    """
    p = np.array(predicted, dtype=float)
    g = np.array(ground_truth, dtype=float)
    if len(p) != len(g) or len(p) < 2:
        return 0.0
    k = min(k, len(p))
    pred_ranking = np.argsort(p)[::-1][:k]
    ideal_ranking = np.argsort(g)[::-1][:k]

    dcg = 0.0
    for i, idx in enumerate(pred_ranking):
        dcg += g[idx] / np.log2(i + 2)

    idcg = 0.0
    for i, idx in enumerate(ideal_ranking):
        idcg += g[idx] / np.log2(i + 2)

    if idcg < 1e-12:
        return 0.0
    return round(float(dcg / idcg), 4)


def ranking_metrics(
    predicted: List[float],
    ground_truth: List[float],
    method_name: str = "",
) -> dict:
    """
    Compute all ranking/correlation metrics between a predicted drift curve
    and a continuous ground-truth drift signal.
    """
    return {
        "method": method_name,
        "spearman": spearman_correlation(predicted, ground_truth),
        "kendall_tau": kendall_tau(predicted, ground_truth),
        "r_squared": r_squared(predicted, ground_truth),
        "ndcg_5": ndcg_at_k(predicted, ground_truth, k=5),
        "ndcg_10": ndcg_at_k(predicted, ground_truth, k=10),
    }


# ── Change-point detection metrics ───────────────────────────────────────

def hit_at_k(drift_curve: List[float], true_idx: int, k: int = 3, tolerance: int = 1) -> bool:
    """
    Is the true change point (within ±tolerance) among the top-K peaks?
    tolerance=1 means index 15, 16, or 17 all count as a hit for true_idx=16.
    """
    arr = np.array(drift_curve)
    top_k_indices = np.argsort(arr)[::-1][:k]
    for idx in top_k_indices:
        if abs(idx - true_idx) <= tolerance:
            return True
    return False


def detection_delay(drift_curve: List[float], true_idx: int) -> int:
    """Absolute distance between the detected peak and the true change point."""
    return abs(int(np.argmax(drift_curve)) - true_idx)


def mean_reciprocal_rank(drift_curve: List[float], true_idx: int, tolerance: int = 1) -> float:
    """
    1/rank of the first peak (within ±tolerance) that matches the true change point.
    MRR=1.0 means the true change point is the #1 detected peak.
    """
    arr = np.array(drift_curve)
    rank_order = np.argsort(arr)[::-1]
    for r, idx in enumerate(rank_order, 1):
        if abs(idx - true_idx) <= tolerance:
            return round(1.0 / r, 4)
    return 0.0


def cpd_precision_recall_f1(
    drift_curve: List[float],
    true_indices: List[int],
    top_k: int = 5,
    tolerance: int = 1,
) -> dict:
    """
    Change-point detection Precision, Recall, and F1.
    Predicted change points = top-K peaks in the drift curve.
    A predicted peak is a true positive if it's within ±tolerance of any true change point.
    """
    arr = np.array(drift_curve)
    predicted = list(np.argsort(arr)[::-1][:top_k])

    tp = 0
    matched_true = set()
    for p in predicted:
        for t in true_indices:
            if abs(p - t) <= tolerance and t not in matched_true:
                tp += 1
                matched_true.add(t)
                break

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(true_indices) if true_indices else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "cpd_precision": round(precision, 4),
        "cpd_recall": round(recall, 4),
        "cpd_f1": round(f1, 4),
        "top_k": top_k,
        "tolerance": tolerance,
    }


# ── Legacy interfaces ────────────────────────────────────────────────────

def compute_learned_drift(
    model: torch.nn.Module,
    persona_vectors: Dict[str, np.ndarray],
    window_ids: List[str],
    device: Optional[str] = None,
) -> List[float]:
    """Old-model interface: project persona vectors, compute cosine distance."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    vecs = np.array([persona_vectors[wid] for wid in window_ids])
    with torch.no_grad():
        proj = model(torch.FloatTensor(vecs).to(device)).cpu().numpy()
    out = []
    for i in range(len(proj) - 1):
        a, b = proj[i], proj[i + 1]
        out.append(float(1.0 - np.dot(a, b)))
    return out


def contrastiveness_separation(
    model: torch.nn.Module,
    X_t: np.ndarray,
    X_t1: np.ndarray,
    y: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Old-model interface: contrastiveness via projected cosine distances."""
    if len(y) < 2 or (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return 0.0, 0.0
    dist_raw = np.array([1 - np.dot(X_t[i], X_t1[i]) for i in range(len(y))])
    sep_before = float(dist_raw[y == 1].mean() - dist_raw[y == 0].mean())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        pt = model(torch.FloatTensor(X_t).to(device)).cpu().numpy()
        pt1 = model(torch.FloatTensor(X_t1).to(device)).cpu().numpy()
    dist_proj = np.array([1 - np.dot(pt[i], pt1[i]) for i in range(len(y))])
    sep_after = float(dist_proj[y == 1].mean() - dist_proj[y == 0].mean())
    return sep_before, sep_after


# ── Full evaluation report ───────────────────────────────────────────────

def full_evaluation_report(
    learned_drift: List[float],
    baseline_sbert: List[float],
    lda_drift: List[float],
    tfidf_drift: List[float],
    mmd_drift: List[float] = None,
    sep_before: float = 0.0,
    sep_after: float = 0.0,
    true_change_idx: Optional[int] = None,
    weak_labels_all: np.ndarray = None,
) -> dict:
    """
    Comprehensive evaluation report with drift-curve, classification, and
    change-point detection metrics.
    """
    corr_sbert = spearman_correlation(baseline_sbert, learned_drift)
    report = {
        "mean_drift_sbert":    round(float(np.mean(baseline_sbert)), 5),
        "mean_drift_lda":      round(float(np.mean(lda_drift)), 5),
        "mean_drift_tfidf":    round(float(np.mean(tfidf_drift)), 5),
        "mean_drift_learned":  round(float(np.mean(learned_drift)), 5),
        "smoothness_sbert":    round(temporal_smoothness(baseline_sbert), 5),
        "smoothness_lda":      round(temporal_smoothness(lda_drift), 5),
        "smoothness_tfidf":    round(temporal_smoothness(tfidf_drift), 5),
        "smoothness_learned":  round(temporal_smoothness(learned_drift), 5),
        "contrastiveness_before": round(sep_before, 5),
        "contrastiveness_after":  round(sep_after, 5),
        "contrastiveness_improved": sep_after > sep_before,
        "spearman_sbert_learned": round(corr_sbert, 5),
        "peak_learned": change_point_rank(learned_drift, true_change_idx),
        "peak_sbert":   change_point_rank(baseline_sbert, true_change_idx),
    }

    if mmd_drift is not None:
        report["mean_drift_mmd"] = round(float(np.mean(mmd_drift)), 5)
        report["smoothness_mmd"] = round(temporal_smoothness(mmd_drift), 5)
        report["peak_mmd"] = change_point_rank(mmd_drift, true_change_idx)
        report["spearman_sbert_mmd"] = round(spearman_correlation(baseline_sbert, mmd_drift), 5)

    # Ranking metrics against ground-truth (if provided)
    if weak_labels_all is not None:
        valid_idx = np.where(weak_labels_all != -1)[0]
 
        if len(valid_idx) > 1:
            gt = weak_labels_all[valid_idx].astype(float)
 
            learned_valid = np.array(learned_drift)[valid_idx]
            sbert_valid = np.array(baseline_sbert)[valid_idx]
            lda_valid = np.array(lda_drift)[valid_idx]
            tfidf_valid = np.array(tfidf_drift)[valid_idx]
 
            report["weak_label_eval_learned"] = ranking_metrics(
                predicted=learned_valid.tolist(),
                ground_truth=gt.tolist(),
                method_name="learned",
            )
            report["weak_label_eval_sbert"] = ranking_metrics(
                predicted=sbert_valid.tolist(),
                ground_truth=gt.tolist(),
                method_name="sbert",
            )
            report["weak_label_eval_lda"] = ranking_metrics(
                predicted=lda_valid.tolist(),
                ground_truth=gt.tolist(),
                method_name="lda",
            )
            report["weak_label_eval_tfidf"] = ranking_metrics(
                predicted=tfidf_valid.tolist(),
                ground_truth=gt.tolist(),
                method_name="tfidf",
            )

            if mmd_drift is not None:
                mmd_valid = np.array(mmd_drift)[valid_idx]
                report["weak_label_eval_mmd"] = ranking_metrics(
                    predicted=mmd_valid.tolist(),
                    ground_truth=gt.tolist(),
                    method_name="mmd",
                )

    # Change-point detection metrics
    if true_change_idx is not None:
        true_indices = [true_change_idx]
        for method_name, curve in [("learned", learned_drift), ("sbert", baseline_sbert)]:
            prefix = f"cpd_{method_name}"
            report[f"{prefix}_hit1"] = hit_at_k(curve, true_change_idx, k=1, tolerance=1)
            report[f"{prefix}_hit3"] = hit_at_k(curve, true_change_idx, k=3, tolerance=1)
            report[f"{prefix}_hit5"] = hit_at_k(curve, true_change_idx, k=5, tolerance=1)
            report[f"{prefix}_delay"] = detection_delay(curve, true_change_idx)
            report[f"{prefix}_mrr"] = mean_reciprocal_rank(curve, true_change_idx, tolerance=1)
            report[f"{prefix}_prf"] = cpd_precision_recall_f1(curve, true_indices, top_k=5, tolerance=1)

    return report
