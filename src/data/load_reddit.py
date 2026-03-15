"""
Load Reddit data from local CSV (e.g. data reddit folder) or from HuggingFace.
If local_data_dir is set and contains dataset_clean.csv or train.csv, that is used (no download).
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

try:
    from pyarrow import ArrowInvalid
except ImportError:
    ArrowInvalid = Exception  # no pyarrow, use Exception


def load_reddit_data_from_local(data_dir: str, min_words: int = 10) -> pd.DataFrame:
    """Load from local CSV(s). Tries dataset_clean.csv, then train.csv, then train+val+test."""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        for base in [Path.cwd(), Path.cwd().parent]:
            if (base / data_dir).exists():
                data_dir = base / data_dir
                break
        else:
            data_dir = Path.cwd() / data_dir
    if not data_dir.exists():
        return pd.DataFrame()
    for name in ["dataset_clean.csv", "train.csv"]:
        path = data_dir / name
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        parts = []
        for name in ["train.csv", "val.csv", "test.csv"]:
            if (data_dir / name).exists():
                parts.append(pd.read_csv(data_dir / name))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty or "text" not in df.columns:
        if "raw_text" in df.columns:
            df["text"] = df["raw_text"].fillna("").astype(str)
        else:
            return pd.DataFrame()
    df["text"] = df["text"].astype(str).str.strip()
    if "created_utc" not in df.columns:
        df["created_utc"] = 0
    df = df[df["text"].str.split().str.len() >= min_words].copy()
    return df


def load_reddit_data(
    dataset_name: str = "HuggingFaceGECLM/REDDIT_comments",
    subreddit_config: str = "todayilearned",
    max_samples: int = 40000,
    min_words: int = 10,
    max_words: int = 512,
    local_data_dir: str = None,
) -> pd.DataFrame:
    """
    Load Reddit data. If local_data_dir is set and has CSV(s), load from disk (no download).
    Returns DataFrame with columns: text, created_utc; optional: window_id.
    """
    if local_data_dir and str(local_data_dir).strip():
        df = load_reddit_data_from_local(local_data_dir, min_words=min_words)
        if not df.empty and len(df) >= 100:
            if "created_utc" not in df.columns:
                df["created_utc"] = 0
            print(f"Loaded local data from '{local_data_dir}': {len(df):,} posts (no download).")
            return df
        print(f"Local data missing or too small in '{local_data_dir}'; using HuggingFace.")
    print(f"Loading: {dataset_name} (r/{subreddit_config}), max {max_samples} rows...")

    try:
        # Try per-subreddit config first (some versions have e.g. "todayilearned", "gaming")
        try:
            ds = load_dataset(
                dataset_name,
                subreddit_config,
                split=f"train[:{max_samples}]",
                trust_remote_code=True,
            )
            filter_subreddit = False
        except Exception as cfg_err:
            msg = str(cfg_err).lower()
            if "not found" in msg or "available" in msg or "default" in msg:
                # Dataset only has "default": use STREAMING to avoid JSON parse errors on corrupt rows
                df = _load_pushshift_default_streaming(
                    dataset_name, subreddit_config, max_samples, min_words, max_words
                )
                if df is not None and len(df) >= 100:
                    if "created_utc" not in df.columns:
                        df["created_utc"] = 0
                    print(f"Loaded {len(df):,} posts (r/{subreddit_config} via streaming).")
                    return df
                # streaming failed or too few rows; try one more fallback below
                raise cfg_err
            else:
                raise cfg_err

        rows = []
        target_sub = subreddit_config.strip().lower()
        for i in tqdm(range(len(ds)), desc="Processing comments"):
            try:
                row = ds[i]
                if filter_subreddit:
                    if str(row.get("subreddit", "")).strip().lower() != target_sub:
                        continue
                body = str(row.get("body", "") or "").strip()
                created = row.get("created_utc")
                if created is None or body in ("[deleted]", "[removed]") or len(body) < 20:
                    continue
                ts = int(created) if isinstance(created, (int, float)) else int(float(str(created)))
                words = body.split()[:max_words]
                if len(words) < min_words:
                    continue
                rows.append({"text": " ".join(words), "created_utc": ts})
                if filter_subreddit and len(rows) >= max_samples:
                    break
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if filter_subreddit:
            print(f"Filtered to r/{subreddit_config}: {len(df):,} posts.")
    except (ValueError, ArrowInvalid, Exception) as e:
        print(f"Dataset load error: {type(e).__name__}: {e}. Falling back to streaming reddit-title-body...")
        df = _load_fallback(subreddit_config, max_samples, min_words, max_words)

    if df.empty or len(df) < 100:
        raise ValueError(f"Too few posts loaded ({len(df)}). Try another subreddit or increase max_samples.")

    if "created_utc" not in df.columns:
        df["created_utc"] = 0
    print(f"Loaded {len(df):,} posts.")
    return df


def _load_pushshift_default_streaming(
    dataset_name: str,
    subreddit: str,
    max_samples: int,
    min_words: int,
    max_words: int,
):
    """
    Load Pushshift (default config) in streaming mode and filter by subreddit.
    Avoids loading full JSON into memory and skips corrupt rows.
    Returns DataFrame or None if streaming not supported / fails.
    """
    try:
        ds = load_dataset(
            dataset_name,
            "default",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        return None
    rows = []
    target_sub = subreddit.strip().lower()
    n_seen = 0
    for item in tqdm(ds, total=None, desc="Streaming Pushshift (default)", leave=True):
        try:
            n_seen += 1
            if str(item.get("subreddit", "")).strip().lower() != target_sub:
                continue
            body = str(item.get("body", "") or "").strip()
            created = item.get("created_utc")
            if created is None or body in ("[deleted]", "[removed]") or len(body) < 20:
                continue
            ts = int(created) if isinstance(created, (int, float)) else int(float(str(created)))
            words = body.split()[:max_words]
            if len(words) < min_words:
                continue
            rows.append({"text": " ".join(words), "created_utc": ts})
            if len(rows) >= max_samples:
                break
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else None


def _load_fallback(
    subreddit: str,
    max_samples: int,
    min_words: int,
    max_words: int,
) -> pd.DataFrame:
    """Fallback: stream from sentence-transformers/reddit-title-body."""
    posts = []
    ds = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)
    for item in tqdm(ds, total=min(80000, max_samples * 3), desc="Streaming"):
        try:
            if str(item.get("subreddit", "")).lower() != subreddit.lower():
                continue
            text = (str(item.get("title", "")) + " " + str(item.get("body", ""))).strip()
            if len(text.split()) < min_words or "[deleted]" in text.lower() or "[removed]" in text.lower():
                continue
            created = item.get("created_utc", 0)
            ts = int(created) if isinstance(created, (int, float)) else 0
            posts.append({"text": " ".join(text.split()[:max_words]), "created_utc": ts})
            if len(posts) >= max_samples:
                break
        except Exception:
            continue
    return pd.DataFrame(posts)
