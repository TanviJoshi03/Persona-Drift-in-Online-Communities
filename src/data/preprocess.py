"""
Temporal segmentation: group posts into monthly windows from created_utc.
Train/test split by time (first train_ratio of windows = train, rest = test).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def temporal_segmentation(
    df: pd.DataFrame,
    min_posts_per_month: int = 200,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Build time windows. If df has a 'window_id' column (e.g. from local saved data),
    use it directly. Otherwise assign from created_utc (monthly) or fixed chunks.
    Returns:
        windows: dict[window_id -> list of text]
        window_ids: ordered list of window ids
    """
    if "window_id" in df.columns:
        # Use pre-computed windows from local data (e.g. data reddit/dataset_clean.csv)
        df_w = df.dropna(subset=["window_id"])
        if len(df_w) >= 100:
            wcounts = df_w.groupby("window_id").size()
            wcounts = wcounts[wcounts >= min_posts_per_month]
            window_ids = [str(w) for w in sorted(wcounts.index.tolist())]
            windows = {wid: df_w[df_w["window_id"].astype(str) == wid]["text"].tolist() for wid in window_ids}
            if len(windows) >= 2:
                return windows, window_ids
    if "created_utc" not in df.columns or df["created_utc"].max() == 0:
        target_windows = max(6, min(80, len(df) // 30000))
        chunk = len(df) // target_windows
        window_ids = [f"W{i+1}" for i in range(target_windows)]
        windows = {
            f"W{i+1}": df.iloc[i * chunk : (i + 1) * chunk]["text"].tolist()
            for i in range(target_windows)
        }
        windows = {k: v for k, v in windows.items() if len(v) >= 50}
        window_ids = list(windows.keys())
        return windows, window_ids

    df = df.copy()
    df["date"] = pd.to_datetime(df["created_utc"], unit="s")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    month_counts = df.groupby("month").size()
    month_counts = month_counts[month_counts >= min_posts_per_month]
    window_ids = sorted(month_counts.index.tolist())
    windows = {wid: df[df["month"] == wid]["text"].tolist() for wid in window_ids}

    if len(window_ids) < 2:
        target_windows = max(6, min(80, len(df) // 30000))
        chunk = len(df) // target_windows
        window_ids = [f"W{i+1}" for i in range(target_windows)]
        windows = {
            f"W{i+1}": df.iloc[i * chunk : (i + 1) * chunk]["text"].tolist()
            for i in range(target_windows)
        }
        windows = {k: v for k, v in windows.items() if len(v) >= 50}
        window_ids = list(windows.keys())

    return windows, window_ids


def train_test_split_windows(
    window_ids: List[str],
    train_ratio: float = 0.75,
) -> Tuple[List[str], List[str]]:
    """Split window_ids into train (first train_ratio) and test (rest)."""
    n_train = max(2, int(len(window_ids) * train_ratio))
    train_ids = window_ids[:n_train]
    test_ids = window_ids[n_train:] if n_train < len(window_ids) else []
    return train_ids, test_ids
