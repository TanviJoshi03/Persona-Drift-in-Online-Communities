# Persona Drift in Online Communities

**CSE 261 Project** — Sara Chaudhari, Tanvi Ganesh Joshi — UC San Diego

Unsupervised detection of persona drift in Reddit communities using a **Multi-View Drift Detector** that fuses neural text embeddings with distributional baselines, trained via weak supervision.

---

## Project Structure

```
persona-drift/
├── config.yaml
├── requirements.txt
├── download_data.py
├── run_pipeline.py
├── PROJECT_PLAN.md
├── README.md
├── data reddit_tech/
│   ├── dataset_clean.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── notebooks/
│   └── final_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_reddit.py
│   │   └── preprocess.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── sbert_encoder.py
│   │   └── baselines.py
│   ├── weak_supervision/
│   │   ├── __init__.py
│   │   └── labels.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   └── contrastive_encoder.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── interpret/
│       ├── __init__.py
│       ├── keywords.py
│       ├── topics.py
│       └── representative_posts.py
└── outputs/
```

---

## File Descriptions

### Root-Level Files

| File | Description |
|---|---|
| `config.yaml` | Settings file for the whole project. You change things here like which subreddit to analyze, how many posts to use, model training settings (learning rate, epochs, etc.), and where to save outputs. Every script and the notebook reads this file. |
| `requirements.txt` | List of all Python libraries the project needs. Run `pip install -r requirements.txt` to install everything. |
| `download_data.py` | Downloads Reddit posts from HuggingFace for a subreddit you choose. You can pick which subreddit (`--subreddit`), how many posts (`--max_posts`), and from what year onwards (`--min_year`). It cleans out deleted/short posts and saves the data as CSV files so you don't have to download again. |
| `run_pipeline.py` | Runs the entire project from the command line (no notebook needed). Does everything: loads data, groups posts by month, computes all drift signals, trains the model, evaluates, makes plots, and saves results to `outputs/`. |
| `PROJECT_PLAN.md` | Project planning document with our task breakdown and design decisions. |
| `README.md` | This file. |

### `data reddit_tech/` — Downloaded Data

| File | Description |
|---|---|
| `dataset_clean.csv` | The full cleaned dataset. Each row is one Reddit post with its text, timestamp, subreddit name, word count, and which split it belongs to. Sorted oldest to newest. This is the main data file the pipeline reads. |
| `train.csv` | The first 75% of posts (earliest in time). Used to train the model and create weak labels. |
| `val.csv` | The next 10% of posts. Used for validation during training. |
| `test.csv` | The last 15% of posts (most recent). Used to evaluate the model on unseen data. |

### `notebooks/`

| File | Description |
|---|---|
| `final_pipeline.ipynb` | **The main notebook — run this.** Goes through the entire pipeline step by step: loads data, groups posts into monthly windows, encodes text, computes drift signals, trains the model, shows evaluation metrics, and generates all the plots (drift curves, heatmaps, scatter plots, radar charts, etc.). |

### `src/data/` — Data Loading & Preprocessing

| File | Description |
|---|---|
| `__init__.py` | Makes the functions in this folder importable by other files. |
| `load_reddit.py` | Reads the Reddit data. First checks if you have data saved locally (CSV files); if yes, loads from there (no internet needed). If not, downloads from HuggingFace. Handles edge cases like missing columns or corrupted rows. |
| `preprocess.py` | Groups posts into time windows. Looks at each post's timestamp and groups them by month (e.g., all Jan 2020 posts = one window, all Feb 2020 posts = another window). If timestamps aren't available, splits the data into equal-sized chunks instead. Also splits window IDs into train/test sets. |

### `src/representation/` — Text Representations & Baselines

| File | Description |
|---|---|
| `__init__.py` | Makes the functions in this folder importable. |
| `sbert_encoder.py` | Converts Reddit posts into numbers (embeddings) using Sentence-BERT. For each monthly window, it encodes all posts and then averages them into a single **persona vector** — one vector that represents "what the community sounds like" that month. |
| `baselines.py` | Computes four different ways to measure how much the community changed between consecutive months: **(1) SBERT** — how different the persona vectors are (cosine distance), **(2) LDA** — how much the discussion topics changed, **(3) TF-IDF** — how much the important keywords changed, **(4) MMD** — how different the full distributions of post embeddings are (catches subtle changes that averages miss). |

### `src/weak_supervision/` — Automatic Labeling

| File | Description |
|---|---|
| `__init__.py` | Makes the functions in this folder importable. |
| `labels.py` | Automatically creates training labels with no manual work. Looks at the SBERT distances between consecutive months and labels the smallest ones as "stable" (community didn't change much) and the largest ones as "shift" (community changed a lot). The model then learns from these labels. |

### `src/model/` — Neural Models

| File | Description |
|---|---|
| `__init__.py` | Makes both models importable. |
| `drift_detector.py` | **The main model we use.** Takes in the persona vectors for two consecutive months plus all four baseline drift signals, and predicts a single "drift score" — how much the community changed. It learns to combine all these signals together (that's the "multi-view fusion"). Trained using both the weak labels and the raw SBERT distances. Has smoothing built in to reduce noise in the output. |
| `contrastive_encoder.py` | **The first model we tried (baseline).** A simpler approach that just projects persona vectors into a new space and measures cosine distance there. We replaced it with the Multi-View Drift Detector because that one gives better results by combining multiple signals instead of relying on just one. Kept here for comparison. |

### `src/evaluation/` — Metrics

| File | Description |
|---|---|
| `__init__.py` | Makes the metric functions importable. |
| `metrics.py` | All the numbers we use to evaluate how well the model works. Includes: **smoothness** (is the drift curve noisy or clean?), **Spearman/Kendall correlation** (do different methods agree on when drift happened?), **R²** (how well does predicted drift match baseline?), **NDCG** (does the model rank the biggest changes correctly?), **peak detection** (can it find the biggest change points?), and **precision/recall/F1** for change-point detection. Also has `full_evaluation_report` which runs all metrics at once and gives you a summary. |

### `src/interpret/` — Interpretability

| File | Description |
|---|---|
| `__init__.py` | Makes the functions in this folder importable. |
| `keywords.py` | Tells you **what words changed** at a detected change point. Compares the text before and after the change and finds words that became more popular ("emerging") or less popular ("declining"). Also does the same for phrases (2-3 word combinations). |
| `topics.py` | Tells you **what topics** a community is talking about in a given month. Uses LDA topic modeling to find clusters of related words (e.g., a "gaming hardware" topic or an "esports" topic). |
| `representative_posts.py` | Finds the **most typical posts** for a given month — the ones whose meaning is closest to the average of all posts that month. Helps you read actual examples of what people were saying. |

### `outputs/`

Empty folder where results get saved: plots (PNG images), metric reports, and trained model files.

---

## Quick Start

### 1. Install

```bash
cd persona-drift
pip install -r requirements.txt
```

### 2. Download data

```bash
python download_data.py --subreddit technology --min_year 2019
```

### 3. Run the final notebook (recommended)

```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

Run all cells in order.

### 4. Or run from command line

```bash
python run_pipeline.py
```

---

## How It Works

### Pipeline

1. **Data**: Load Pushshift Reddit comments (HuggingFace or local CSV), build monthly temporal windows.
2. **Representation**: Compute four baseline drift signals between consecutive windows:
   - **SBERT**: Cosine distance between mean persona vectors
   - **LDA**: Jensen-Shannon divergence of topic distributions
   - **TF-IDF**: Jensen-Shannon divergence of keyword vectors
   - **MMD**: Maximum Mean Discrepancy between full embedding distributions (captures distributional shifts beyond means)
3. **Weak supervision**: Label window transitions as "stable" or "shift" using percentile thresholds on SBERT distances (train windows only). No manual labels required.
4. **Multi-View Drift Detector**: Neural fusion model that:
   - Encodes persona-vector differences and element-wise interactions
   - Fuses with all four baseline signals
   - Predicts a drift score per transition
   - Trained on ALL transitions (regression on SBERT distances) + contrastive loss on labeled pairs
   - Post-processing: Gaussian temporal smoothing
5. **Evaluation**: Drift-curve quality (smoothness, peak height), inter-method agreement (Spearman, Kendall, R², NDCG), contrastiveness improvement, and unsupervised change-point discovery.
6. **Interpretability**: Representative posts, keyword/phrase shifts, and LDA topics at the top detected change points.

### What Makes This Novel

- **Multi-view fusion**: The detector jointly reasons over learned semantic features and four complementary drift signals, producing more robust change-point detection than any single method.
- **MMD baseline**: Compares full embedding distributions (not just means), capturing higher-order distributional differences.
- **Training on all transitions**: Uses continuous regression targets for all window pairs, not just binary labels on a small subset.
- **Weak supervision framework**: No manual labels needed; stable/shift pairs derived automatically from percentile thresholds.
- **Temporal smoothing**: Reduces noise in the final drift curve while preserving real peaks.

---

## Configuration

Edit **`config.yaml`** to change:

- **data**: `local_data_dir`, `dataset_name`, `subreddit_config`, `max_samples`, `min_posts_per_month`
- **split**: `train_ratio` (temporal holdout, default 0.75)
- **representation**: `sbert_model`, `lda_n_topics`, `tfidf_max_features`
- **weak_supervision**: `stable_percentile`, `shift_percentile`
- **detector**: `persona_dim`, `n_baselines`, `proj_dim`, `hidden_dim`, `dropout`, `learning_rate`, `epochs`, `patience`, `contrastive_weight`, `gradient_clip`, `smooth_sigma`
- **output**: `save_dir`, `save_artifacts`

---

## Data: Local or HuggingFace

- **Download first (recommended):** Run `python download_data.py --subreddit technology --min_year 2019` to save data locally, then point `data.local_data_dir` in `config.yaml` to the saved folder.
- **Local (no download):** Place Reddit data in a folder with `dataset_clean.csv` (or `train.csv`). Set `data.local_data_dir` in `config.yaml`.
- **HuggingFace (on-the-fly):** If `local_data_dir` is empty or missing, data is streamed from `HuggingFaceGECLM/REDDIT_comments`.

---

## Interpreting Evaluation Metrics

| Metric | Bad | Good |
|---|---|---|
| **Contrastiveness (after > before)** | No improvement | Yes, and after > 0.10 |
| **Spearman(SBERT, Learned)** | < 0.3 | > 0.70 |
| **Kendall tau** | < 0.2 | > 0.50 |
| **R²** | < 0.0 (negative) | > 0.50 |
| **NDCG@5** | < 0.5 | > 0.80 |
| **Peak height ratio** | < 1.5 | > 3.0 (peak much higher than average) |
| **Temporal smoothness** | 0.0 (flat) or very high | Similar to SBERT baseline |

---

**Based on CSE 261 Project Proposal** — Persona drift detection with weak supervision and neural multi-view fusion.
