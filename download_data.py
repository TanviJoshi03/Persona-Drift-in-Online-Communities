"""
Download Reddit data from HuggingFace for a SINGLE subreddit and save locally.

The pipeline then performs persona drift analysis on the real temporal
evolution of that community's language — no synthetic mixing.

Temporal segmentation (monthly windows) is handled by the pipeline itself
using created_utc timestamps.

Output files in DATA_DIR:
  - dataset_clean.csv   (text, created_utc, subreddit, n_words)
  - train.csv / val.csv / test.csv  (temporal split)

Usage:
    python download_data.py                                    # default: technology, 2019+
    python download_data.py --subreddit gaming                 # choose a subreddit
    python download_data.py --subreddit gaming --min_year 2020 # only 2020+ data
    python download_data.py --subreddit movies --max_posts 500000
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("data reddit")
DEFAULT_SUBREDDIT = "technology"
DEFAULT_MAX_POSTS = 3000000
MIN_WORDS = 10
DEFAULT_MIN_YEAR = 2019


PUSHSHIFT_SUBREDDITS = [
    "AskHistorians", "DIY", "Damnthatsinteresting", "Documentaries",
    "EatCheapAndHealthy", "Fantasy", "Fitness", "Foodforthought", "Games",
    "GetMotivated", "IAmA", "IWantToLearn", "LifeProTips", "Showerthoughts",
    "SkincareAddiction", "UpliftingNews", "WritingPrompts", "YouShouldKnow",
    "askscience", "bestof", "boardgames", "bodyweightfitness", "books",
    "buildapc", "changemyview", "explainlikeimfive", "femalefashionadvice",
    "gadgets", "gaming", "gardening", "history", "ifyoulikeblank", "lifehacks",
    "malefashionadvice", "mildlyinteresting", "personalfinance", "philosophy",
    "podcasts", "programming", "relationship_advice", "science", "scifi",
    "socialskills", "space", "sports", "suggestmeabook", "technology", "tifu",
    "todayilearned", "travel",
]


def _find_pushshift_split(name: str) -> str:
    """Find the matching Pushshift split name (case-insensitive)."""
    target = name.strip().lower()
    for s in PUSHSHIFT_SUBREDDITS:
        if s.lower() == target:
            return s
    return ""


def download_subreddit(name: str, max_posts: int, min_year: int = 2019) -> pd.DataFrame:
    """Download posts for a subreddit from HuggingFace, filtering by year."""
    from datasets import load_dataset
    import datetime

    target = name.strip().lower()
    posts = []
    skipped_old = 0

    min_ts = int(datetime.datetime(min_year, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    print("Filtering: only keeping posts from {} onwards (unix >= {})".format(min_year, min_ts))

    # ── Source 1: Pushshift (subreddit = split name) ─────────────────
    split_name = _find_pushshift_split(name)
    if split_name:
        print("Downloading from HuggingFaceGECLM/REDDIT_comments (split='{}') ...".format(split_name))
        try:
            ds = load_dataset(
                "HuggingFaceGECLM/REDDIT_comments",
                split=split_name,
                streaming=True,
            )
            for item in tqdm(ds, desc="Streaming r/{}".format(name), leave=True):
                try:
                    body = str(item.get("body", "") or "").strip()
                    words = body.split()
                    if len(words) < MIN_WORDS or body in ("[deleted]", "[removed]"):
                        continue
                    text = " ".join(words)
                    created = item.get("created_utc", 0)
                    try:
                        ts = int(float(created)) if created else 0
                    except (ValueError, TypeError):
                        ts = 0
                    if ts < min_ts:
                        skipped_old += 1
                        if skipped_old % 100000 == 0:
                            print("  ... skipped {:,} pre-{} posts so far".format(skipped_old, min_year))
                        continue
                    posts.append({
                        "subreddit": name,
                        "text": text,
                        "created_utc": ts,
                    })
                    if len(posts) >= max_posts:
                        break
                    if len(posts) % 50000 == 0 and len(posts) > 0:
                        print("  ... {:,} posts collected (skipped {:,} old)".format(len(posts), skipped_old))
                except Exception:
                    continue
            print("Got {:,} posts from Pushshift (skipped {:,} pre-{})".format(len(posts), skipped_old, min_year))
        except Exception as e:
            print("Pushshift failed: {}".format(e))
    else:
        print("r/{} not available in Pushshift. Available subreddits:".format(name))
        print("  {}".format(", ".join(PUSHSHIFT_SUBREDDITS)))

    # ── Source 2: sentence-transformers/reddit-title-body (fallback) ──
    if len(posts) < max_posts // 2:
        print("Trying sentence-transformers/reddit-title-body as fallback ...")
        try:
            ds2 = load_dataset(
                "sentence-transformers/reddit-title-body",
                split="train",
                streaming=True,
            )
            seen = 0
            for item in tqdm(ds2, desc="Fallback r/{}".format(name), leave=True):
                seen += 1
                try:
                    sub = str(item.get("subreddit", "")).strip().lower()
                    if sub != target:
                        if seen > 5_000_000 and len(posts) < 50:
                            break
                        continue
                    title = str(item.get("title", "") or "").strip()
                    body = str(item.get("body", "") or "").strip()
                    text = (title + " " + body).strip()
                    words = text.split()
                    if len(words) < MIN_WORDS:
                        continue
                    if "[deleted]" in text or "[removed]" in text:
                        continue
                    created = item.get("created_utc", 0)
                    try:
                        ts = int(float(created)) if created else 0
                    except (ValueError, TypeError):
                        ts = 0
                    if ts < min_ts:
                        skipped_old += 1
                        continue
                    posts.append({
                        "subreddit": name,
                        "text": text,
                        "created_utc": ts,
                    })
                    if len(posts) >= max_posts:
                        break
                except Exception:
                    continue
            print("Total after fallback: {:,} posts".format(len(posts)))
        except Exception as e:
            print("Fallback also failed: {}".format(e))

    return pd.DataFrame(posts)


def save_dataset(df: pd.DataFrame, subreddit: str):
    """Save to CSV files. Split is by time (first 75% train, next 10% val, rest test)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp so temporal segmentation works correctly
    df = df.sort_values("created_utc").reset_index(drop=True)
    df["n_words"] = df["text"].str.split().str.len()

    # Temporal train/val/test split (75/10/15)
    n = len(df)
    n_train = int(n * 0.75)
    n_val = int(n * 0.10)
    df["split"] = "test"
    df.loc[:n_train - 1, "split"] = "train"
    df.loc[n_train:n_train + n_val - 1, "split"] = "val"

    # Add datetime for readability
    if df["created_utc"].max() > 0:
        df["datetime_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df["year_month"] = df["datetime_utc"].dt.to_period("M").astype(str)
    else:
        df["datetime_utc"] = ""
        df["year_month"] = ""

    # Remove old files first
    for f in DATA_DIR.glob("*.csv"):
        f.unlink()

    df.to_csv(DATA_DIR / "dataset_clean.csv", index=False)
    df[df["split"] == "train"].to_csv(DATA_DIR / "train.csv", index=False)
    df[df["split"] == "val"].to_csv(DATA_DIR / "val.csv", index=False)
    df[df["split"] == "test"].to_csv(DATA_DIR / "test.csv", index=False)

    print("\nSaved to '{}/':".format(DATA_DIR))
    print("  dataset_clean.csv  : {:,} rows".format(len(df)))
    print("  train.csv          : {:,} rows".format((df["split"] == "train").sum()))
    print("  val.csv            : {:,} rows".format((df["split"] == "val").sum()))
    print("  test.csv           : {:,} rows".format((df["split"] == "test").sum()))

    if df["created_utc"].max() > 0:
        date_min = df["datetime_utc"].min()
        date_max = df["datetime_utc"].max()
        n_months = df["year_month"].nunique()
        print("\n  Date range: {} to {}".format(date_min, date_max))
        print("  Unique months: {} (these become temporal windows)".format(n_months))


def main():
    parser = argparse.ArgumentParser(description="Download Reddit data for persona drift analysis")
    parser.add_argument("--subreddit", type=str, default=DEFAULT_SUBREDDIT,
                        help="Subreddit to download (default: {})".format(DEFAULT_SUBREDDIT))
    parser.add_argument("--max_posts", type=int, default=DEFAULT_MAX_POSTS,
                        help="Maximum posts to download (default: {:,})".format(DEFAULT_MAX_POSTS))
    parser.add_argument("--min_year", type=int, default=DEFAULT_MIN_YEAR,
                        help="Only keep posts from this year onwards (default: {})".format(DEFAULT_MIN_YEAR))
    args = parser.parse_args()

    print("=" * 60)
    print("Persona Drift — Data Download")
    print("Subreddit: r/{}".format(args.subreddit))
    print("Target posts: {:,}".format(args.max_posts))
    print("Year filter: {} onwards".format(args.min_year))
    print("=" * 60)

    df = download_subreddit(args.subreddit, args.max_posts, min_year=args.min_year)

    if len(df) < 500:
        print("\nERROR: Only {} posts downloaded for r/{}. Not enough data.".format(
            len(df), args.subreddit))
        print("Try a more popular subreddit or increase --max_posts.")
        sys.exit(1)

    print("\nDownloaded {:,} posts for r/{}".format(len(df), args.subreddit))
    save_dataset(df, args.subreddit)

    print("\nDone! Run the notebook to perform persona drift analysis.")
    print("Data is saved locally — no re-download needed next time.")


if __name__ == "__main__":
    main()
