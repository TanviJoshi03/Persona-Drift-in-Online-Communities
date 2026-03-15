#!/usr/bin/env python3
"""
Run the full Persona Drift pipeline from config.
Usage: python run_pipeline.py [--config config.yaml]
"""
import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_reddit_data, temporal_segmentation, train_test_split_windows
from src.representation import get_persona_vectors, compute_lda_drift, compute_tfidf_drift, compute_mmd_drift
from src.representation.baselines import compute_sbert_baseline_drift
from src.weak_supervision import compute_distances, get_weak_labels
from src.model import MultiViewDriftDetector, train_drift_detector, predict_drift, detector_contrastiveness
from src.evaluation import full_evaluation_report, change_point_rank, spearman_correlation
from src.interpret import keyword_shifts, phrase_shifts, get_lda_topics, get_representative_posts


def load_config(path: str = None) -> dict:
    path = path or str(ROOT / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg_path: str = None) -> dict:
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    repr_cfg = cfg["representation"]
    ws_cfg = cfg["weak_supervision"]
    det_cfg = cfg["detector"]
    out_cfg = cfg.get("output", {})
    save_dir = Path(out_cfg.get("save_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Data
    df = load_reddit_data(
        dataset_name=data_cfg["dataset_name"],
        subreddit_config=data_cfg["subreddit_config"],
        max_samples=data_cfg["max_samples"],
        min_words=data_cfg["min_words_per_post"],
        max_words=data_cfg.get("max_words_per_post", 999999),
        local_data_dir=data_cfg.get("local_data_dir") or "",
    )
    windows, window_ids = temporal_segmentation(df, min_posts_per_month=data_cfg["min_posts_per_month"])
    train_window_ids, test_window_ids = train_test_split_windows(window_ids, train_ratio=split_cfg["train_ratio"])
    print(f"Windows: {len(window_ids)} (train {len(train_window_ids)}, test {len(test_window_ids)})")

    # 2. Representation (4 baselines)
    window_embeddings, persona_vectors, sbert_model = get_persona_vectors(
    windows,
    window_ids,
    model_name=repr_cfg["sbert_model"],
    max_posts_per_window=1000,
    batch_size=32,
)
    baseline_sbert = compute_sbert_baseline_drift(persona_vectors, window_ids)
    lda_drift = compute_lda_drift(windows, window_ids, n_topics=repr_cfg["lda_n_topics"])
    tfidf_drift = compute_tfidf_drift(windows, window_ids, max_features=repr_cfg["tfidf_max_features"])
    mmd_drift = compute_mmd_drift(window_embeddings, window_ids)

    # 3. Weak supervision
    distances, pairs = compute_distances(persona_vectors, window_ids)
    X_t, X_t1, y, raw_dist, pair_ids = get_weak_labels(
        distances, pairs, train_window_ids,
        stable_percentile=ws_cfg["stable_percentile"],
        shift_percentile=ws_cfg["shift_percentile"],
    )
    n_trans = len(window_ids) - 1
    all_weak_labels = np.full(n_trans, -1, dtype=np.int64)
    transition_lookup = {(window_ids[i], window_ids[i + 1]): i for i in range(n_trans)}
    for k, (wid_a, wid_b) in enumerate(pair_ids):
        idx = transition_lookup.get((wid_a, wid_b))
        if idx is not None:
            all_weak_labels[idx] = y[k]
    print(f"Labeled pairs: stable={(y==0).sum()}, shift={(y==1).sum()}")

    # 4. Multi-View Drift Detector
    detector = MultiViewDriftDetector(
        persona_dim=det_cfg["persona_dim"],
        n_baselines=det_cfg["n_baselines"],
        proj_dim=det_cfg["proj_dim"],
        hidden_dim=det_cfg["hidden_dim"],
        dropout=det_cfg["dropout"],
    )
    train_losses = train_drift_detector(
        detector,
        persona_vectors=persona_vectors,
        window_ids=window_ids,
        sbert_drift=baseline_sbert,
        lda_drift=lda_drift,
        tfidf_drift=tfidf_drift,
        mmd_drift=mmd_drift,
        weak_labels=all_weak_labels,
        epochs=det_cfg["epochs"],
        lr=det_cfg["learning_rate"],
        patience=det_cfg["patience"],
        contrastive_weight=det_cfg["contrastive_weight"],
        gradient_clip=det_cfg["gradient_clip"],
        device=device,
    )
    learned_drift = predict_drift(
        detector, persona_vectors, window_ids,
        baseline_sbert, lda_drift, tfidf_drift, mmd_drift,
        smooth_sigma=det_cfg.get("smooth_sigma", 0.8),
        device=device,
    )

    # Debug stats
    print("\n--- Debug stats ---")
    print(f"Num windows: {len(window_ids)}")
    print(f"Num transitions: {len(baseline_sbert)}")

    print(f"SBERT drift -> min: {np.min(baseline_sbert):.6f}, max: {np.max(baseline_sbert):.6f}, std: {np.std(baseline_sbert):.6f}")
    print(f"LDA drift   -> min: {np.min(lda_drift):.6f}, max: {np.max(lda_drift):.6f}, std: {np.std(lda_drift):.6f}")
    print(f"TFIDF drift -> min: {np.min(tfidf_drift):.6f}, max: {np.max(tfidf_drift):.6f}, std: {np.std(tfidf_drift):.6f}")
    print(f"MMD drift   -> min: {np.min(mmd_drift):.6f}, max: {np.max(mmd_drift):.6f}, std: {np.std(mmd_drift):.6f}")
    print(f"Learned     -> min: {np.min(learned_drift):.6f}, max: {np.max(learned_drift):.6f}, std: {np.std(learned_drift):.6f}")

    print(f"Weak labels counts -> stable={(all_weak_labels == 0).sum()}, shift={(all_weak_labels == 1).sum()}, unlabeled={(all_weak_labels == -1).sum()}")

    # 5. Evaluation
    sep_before, sep_after = detector_contrastiveness(
        detector, persona_vectors, window_ids,
        baseline_sbert, lda_drift, tfidf_drift, mmd_drift,
        all_weak_labels, device=device,
    )
    report = full_evaluation_report(
    learned_drift=learned_drift,
    baseline_sbert=baseline_sbert,
    lda_drift=lda_drift,
    tfidf_drift=tfidf_drift,
    mmd_drift=mmd_drift,
    sep_before=sep_before,
    sep_after=sep_after,
    true_change_idx=None,
    weak_labels_all=all_weak_labels,
    )
    print("\n--- Metrics ---")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 6. Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    x = range(len(baseline_sbert))
    ax.plot(x, baseline_sbert, "o-", lw=2, ms=5, label="SBERT baseline", color="steelblue")
    ax.plot(x, lda_drift, "^-", lw=1.5, ms=4, label="LDA (JSD)", color="green", alpha=0.7)
    ax.plot(x, tfidf_drift, "d-", lw=1.5, ms=4, label="TF-IDF (JSD)", color="purple", alpha=0.7)
    ax.plot(x, mmd_drift, "x-", lw=1.5, ms=5, label="MMD (RBF)", color="brown", alpha=0.7)
    ax.plot(x, learned_drift, "s-", lw=2.5, ms=7, label="Learned (Multi-View)", color="darkorange")
    if len(train_window_ids) < len(window_ids):
        ax.axvline(x=len(train_window_ids) - 1.5, color="gray", ls="--", alpha=0.6, label="Train|Test")
    ax.axvline(x=16, color="red", ls=":", alpha=0.5, lw=2, label="True change (idx 16)")
    ax.set_xlabel("Time window transition")
    ax.set_ylabel("Drift value")
    ax.set_title(f"Persona drift: r/{data_cfg['subreddit_config']} — All baselines vs Multi-View Detector")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{window_ids[i]}→{window_ids[i+1]}" for i in x], rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / "drift_curves.png", dpi=150)
    plt.close()

    # 7. Interpretability at top change point
    top_idx = int(np.argmax(learned_drift))
    w_before = window_ids[top_idx]
    w_after = window_ids[top_idx + 1]
    rep_before = get_representative_posts(
        window_embeddings[w_before], persona_vectors[w_before], windows[w_before], top_n=3
    )
    rep_after = get_representative_posts(
        window_embeddings[w_after], persona_vectors[w_after], windows[w_after], top_n=3
    )
    emerging_kw, declining_kw = keyword_shifts(windows[w_before], windows[w_after])
    emerging_ph, declining_ph = phrase_shifts(windows[w_before], windows[w_after])
    topics_before = get_lda_topics(windows[w_before], n_topics=3, top_n_words=5)
    topics_after = get_lda_topics(windows[w_after], n_topics=3, top_n_words=5)

    if out_cfg.get("save_artifacts"):
        torch.save(detector.state_dict(), save_dir / "detector.pt")
        np.savez(
            save_dir / "drift_curves.npz",
            baseline_sbert=baseline_sbert, lda_drift=lda_drift,
            tfidf_drift=tfidf_drift, mmd_drift=mmd_drift,
            learned_drift=learned_drift, window_ids=window_ids,
        )
        pd.DataFrame([{k: v for k, v in report.items() if isinstance(v, (int, float, bool))}]).to_csv(
            save_dir / "metrics.csv", index=False
        )
        with open(save_dir / "interpret.txt", "w") as f:
            f.write(f"Top change: {w_before} -> {w_after}\n\n")
            f.write("Representative before:\n" + "\n".join(rep_before[:3]) + "\n\n")
            f.write("Representative after:\n" + "\n".join(rep_after[:3]) + "\n\n")
            f.write("Emerging keywords: " + str(emerging_kw[:10]) + "\n")
            f.write("Declining keywords: " + str(declining_kw[:10]) + "\n")
            f.write("Topics before: " + str(topics_before) + "\n")
            f.write("Topics after: " + str(topics_after) + "\n")

    return {
        "windows": windows, "window_ids": window_ids,
        "train_window_ids": train_window_ids, "test_window_ids": test_window_ids,
        "persona_vectors": persona_vectors, "window_embeddings": window_embeddings,
        "detector": detector,
        "baseline_sbert": baseline_sbert, "lda_drift": lda_drift,
        "tfidf_drift": tfidf_drift, "mmd_drift": mmd_drift,
        "learned_drift": learned_drift,
        "report": report, "train_losses": train_losses,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to config.yaml")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
