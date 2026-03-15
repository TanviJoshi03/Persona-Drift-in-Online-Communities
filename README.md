# Persona Drift in Online Communities

**CSE 261 Project** — Sara Chaudhari, Tanvi Ganesh Joshi — UC San Diego

Unsupervised detection of persona drift in Reddit communities using a **Multi-View Drift Detector** that fuses neural text embeddings with distributional baselines, trained via weak supervision (no manual labels required).

---

## Overview

Online communities evolve over time — their language, topics, and tone shift in ways that are hard to track manually. This project builds a pipeline that:

1. Collects Reddit posts from a chosen subreddit
2. Groups them into monthly time windows
3. Computes four complementary drift signals (SBERT, LDA, TF-IDF, MMD)
4. Trains a neural fusion model to combine these signals into a single drift score
5. Identifies when and why major persona shifts occurred

---

## Project Structure

```
Persona-Drift-in-Online-Communities-main/
├── config.yaml                  # Central configuration for the entire pipeline
├── requirements.txt             # Python dependencies
├── download_data.py             # Downloads and cleans Reddit data from HuggingFace
├── run_pipeline.py              # Runs the full pipeline from the command line
├── README.md
│
├── notebooks/
│   └── final_pipeline.ipynb     # Interactive notebook that runs the pipeline step-by-step
│
├── src/
│   ├── data/
│   │   ├── load_reddit.py       # Loads Reddit data from local CSV or HuggingFace
│   │   └── preprocess.py        # Groups posts into monthly windows, splits train/test
│   ├── representation/
│   │   ├── sbert_encoder.py     # Encodes posts with Sentence-BERT into persona vectors
│   │   └── baselines.py         # Computes SBERT, LDA, TF-IDF, and MMD drift signals
│   ├── weak_supervision/
│   │   └── labels.py            # Auto-labels transitions as stable/shift using percentiles
│   ├── model/
│   │   ├── drift_detector.py    # Multi-View Drift Detector (main model)
│   │   └── contrastive_encoder.py  # Simpler baseline model (kept for comparison)
│   ├── evaluation/
│   │   └── metrics.py           # Smoothness, correlation, NDCG, change-point F1, etc.
│   └── interpret/
│       ├── keywords.py          # Emerging/declining keywords and phrases at change points
│       ├── topics.py            # LDA topic modeling per window
│       └── representative_posts.py  # Finds posts closest to the window's persona vector
│
└── output/                      # Generated results (after running the pipeline)
    ├── metrics.csv              # Evaluation metrics summary
    ├── interpret.txt            # Top change points with keywords, topics, example posts
    ├── detector.pt              # Trained model weights
    ├── drift_curves.png         # Visualization of all drift curves
    └── drift_curves.npz         # Raw drift curve data (NumPy archive)
```

---

## File Descriptions

### Root-Level Scripts

| File | Description |
|---|---|
| `config.yaml` | Central configuration file. Controls data source, subreddit selection, model hyperparameters (learning rate, epochs, dropout, etc.), representation settings, weak supervision thresholds, and output paths. Every script and the notebook reads this file. |
| `requirements.txt` | Lists all Python dependencies with minimum versions. |
| `download_data.py` | Downloads Reddit posts from HuggingFace for a chosen subreddit. Accepts CLI arguments: `--subreddit` (which community), `--max_posts` (limit), and `--min_year` (earliest year). Cleans out deleted/short posts and saves `dataset_clean.csv`, `train.csv`, `val.csv`, and `test.csv`. |
| `run_pipeline.py` | Runs the entire pipeline end-to-end from the command line: loads data, segments by month, computes drift signals, trains the model, evaluates, generates plots, and saves everything to `output/`. |

### `src/data/` — Data Loading and Preprocessing

| File | Description |
|---|---|
| `load_reddit.py` | Loads Reddit data from local CSV files if available, otherwise downloads from HuggingFace. Handles missing columns, corrupted rows, and fallback datasets. |
| `preprocess.py` | Groups posts into temporal windows by month using timestamps. Falls back to fixed-size chunks if timestamps are unavailable. Also provides train/test splitting of window IDs. |

### `src/representation/` — Text Representations and Drift Signals

| File | Description |
|---|---|
| `sbert_encoder.py` | Encodes posts using Sentence-BERT and averages embeddings per window into a single **persona vector** — a dense representation of what the community sounds like in a given month. |
| `baselines.py` | Computes four drift signals between consecutive windows: **(1) SBERT** — cosine distance between persona vectors, **(2) LDA** — Jensen-Shannon divergence of topic distributions, **(3) TF-IDF** — Jensen-Shannon divergence of keyword importance vectors, **(4) MMD** — Maximum Mean Discrepancy between full embedding distributions, capturing shifts that mean vectors miss. |

### `src/weak_supervision/` — Automatic Labeling

| File | Description |
|---|---|
| `labels.py` | Generates training labels without manual annotation. Computes cosine distances between consecutive persona vectors, then labels the lowest distances as "stable" (0) and highest as "shift" (1) using percentile thresholds. Middle-range transitions are left unlabeled. |

### `src/model/` — Neural Models

| File | Description |
|---|---|
| `drift_detector.py` | **Main model.** The Multi-View Drift Detector takes persona-vector differences and all four baseline drift signals as input, fuses them through learned layers, and outputs a single drift score per transition. Trained with regression loss (on SBERT distances) plus contrastive loss (on weak labels). Supports optional Gaussian smoothing on the output curve. |
| `contrastive_encoder.py` | Baseline model for comparison. Projects persona vectors into a learned space and uses cosine distance. Superseded by the Multi-View Drift Detector which combines multiple signals for better results. |

### `src/evaluation/` — Metrics

| File | Description |
|---|---|
| `metrics.py` | Computes evaluation metrics: **temporal smoothness**, **Spearman/Kendall correlation** between methods, **R²**, **NDCG** for ranking top changes, **peak detection**, and **change-point precision/recall/F1**. The `full_evaluation_report()` function runs all metrics and returns a summary. |

### `src/interpret/` — Interpretability

| File | Description |
|---|---|
| `keywords.py` | Identifies what changed at detected change points by comparing TF-IDF vectors before and after. Reports emerging keywords (gained importance) and declining keywords (lost importance), plus bigram/trigram phrase shifts. |
| `topics.py` | Runs LDA topic modeling on posts within a window to surface the main discussion topics (clusters of related words). |
| `representative_posts.py` | Finds the most typical posts for a given window — those whose embeddings are closest to the window's persona vector. Useful for reading concrete examples of community discourse. |

### `notebooks/`

| File | Description |
|---|---|
| `final_pipeline.ipynb` | Interactive notebook that runs the full pipeline step-by-step: data loading, temporal segmentation, representation, weak labeling, model training, evaluation, and visualization (drift curves, heatmaps, scatter plots). **Recommended way to explore the project.** |

### `output/` — Generated Artifacts

| File | Description |
|---|---|
| `metrics.csv` | Evaluation metrics (mean drift, smoothness, contrastiveness, Spearman, R², etc.). |
| `interpret.txt` | Top change points with representative posts, emerging/declining keywords, and LDA topics. |
| `detector.pt` | Saved weights of the trained Multi-View Drift Detector. |
| `drift_curves.png` | Plot comparing all drift curves (SBERT, LDA, TF-IDF, MMD, learned). |
| `drift_curves.npz` | NumPy archive containing raw drift curve arrays and window IDs. |

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
cd Persona-Drift-in-Online-Communities-main
pip install -r requirements.txt
```

### Download Data

Download Reddit posts for your chosen subreddit:

```bash
python download_data.py --subreddit technology --min_year 2019
```

This creates a `data reddit/` folder containing:
- `dataset_clean.csv` — full cleaned dataset (columns: `text`, `created_utc`, `subreddit`, `n_words`, `split`, `datetime_utc`, `year_month`)
- `train.csv` — first 75% of posts (by time)
- `val.csv` — next 10%
- `test.csv` — last 15%

You can customize the download:

```bash
python download_data.py --subreddit gaming --max_posts 500000 --min_year 2020
```

---

## Running the Pipeline

### Option A: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

Run all cells in order. The notebook walks through each stage with inline visualizations and explanations.

### Option B: Command Line

```bash
python run_pipeline.py
```

Or with a custom config:

```bash
python run_pipeline.py --config config.yaml
```

Results are saved to the `output/` folder.

---

## How It Works

1. **Data** — Load Reddit posts (from local CSV or HuggingFace) and group them into monthly temporal windows.
2. **Representation** — Encode posts with Sentence-BERT and compute four drift signals between consecutive windows:
   - **SBERT**: cosine distance between mean persona vectors
   - **LDA**: Jensen-Shannon divergence of topic distributions
   - **TF-IDF**: Jensen-Shannon divergence of keyword vectors
   - **MMD**: Maximum Mean Discrepancy between full embedding distributions
3. **Weak Supervision** — Automatically label transitions as "stable" or "shift" using percentile thresholds on SBERT distances. No manual annotation needed.
4. **Multi-View Drift Detector** — Train a neural fusion model that combines persona-vector differences with all four baseline signals, producing a single drift score per transition. Uses both regression and contrastive losses.
5. **Evaluation** — Measure drift-curve quality (smoothness, peak height), inter-method agreement (Spearman, Kendall, R², NDCG), and change-point detection accuracy (precision, recall, F1).
6. **Interpretability** — At detected change points, surface representative posts, emerging/declining keywords, and LDA topics to explain *why* the community changed.

---

## Configuration

Edit `config.yaml` to customize the pipeline:

| Section | Key Settings |
|---|---|
| `data` | `local_data_dir` (path to CSV data), `subreddit_config`, `max_samples`, `min_posts_per_month` |
| `split` | `train_ratio` (default 0.75) |
| `representation` | `sbert_model`, `lda_n_topics`, `tfidf_max_features` |
| `weak_supervision` | `stable_percentile`, `shift_percentile` |
| `detector` | `persona_dim`, `proj_dim`, `hidden_dim`, `dropout`, `learning_rate`, `epochs`, `patience`, `contrastive_weight`, `smooth_sigma` |
| `output` | `save_dir`, `save_artifacts` |

---

## Data Sources

- **Local (recommended)**: Run `download_data.py` first, then the pipeline reads from the local `data reddit/` folder.
- **HuggingFace (on-the-fly)**: If no local data is found, posts are streamed from `HuggingFaceGECLM/REDDIT_comments` (Pushshift archive).

Posts are filtered to exclude `[deleted]`/`[removed]` content and posts with fewer than 10 words (configurable via `min_words_per_post`).

---

## Interpreting Evaluation Metrics

| Metric | Weak | Strong |
|---|---|---|
| Contrastiveness (after vs. before) | No improvement | after > before, and after > 0.10 |
| Spearman (SBERT vs. Learned) | < 0.3 | > 0.70 |
| Kendall tau | < 0.2 | > 0.50 |
| R² | < 0.0 (negative) | > 0.50 |
| NDCG@5 | < 0.5 | > 0.80 |
| Peak height ratio | < 1.5 | > 3.0 |
| Temporal smoothness | 0.0 (flat) or very high | Similar to SBERT baseline |


