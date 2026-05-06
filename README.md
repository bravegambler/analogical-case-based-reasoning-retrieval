# Capstone_Context: Nasdaq-100 Multimodal Retrieval Pipeline

A Case-Based Reasoning (CBR) pipeline that retrieves historically similar Nasdaq-100 stock episodes using contrastive-learned BGE-M3 / Contriever embeddings over LLM-summarized 5-day news windows, benchmarked against a DTW price-series ground truth.

## Project Structure

```text
Capstone_Context/
├── config.yaml                      # Central configuration (paths, hyperparams, model settings)
├── main.py                          # Unified entry point for all pipeline stages
├── utility.py                       # Shared utilities (config loading, date parsing, data I/O)
├── models/
│   ├── Retriever.py                 # GPU-accelerated vector & DTW retrieval logic
│   ├── ContrastiveTrainer.py        # SentenceTransformer fine-tuning (agg / sum branches)
│   ├── VectorBasedTrainer.py        # Vector-pooling fine-tuning (pool branch)
│   ├── Poolers.py                   # Attention/mean pooling heads
│   ├── train_metrics.py             # EpochEvaluator + MetricsWriter (per-epoch recall / NDCG)
│   └── training_heartbeat.py        # Heartbeat writer for run_status.txt
├── pipeline/
│   ├── step02_preprocess.py         # Data cleaning, anomaly splitting, news matching
│   ├── step03_dtw_summarize.py      # DTW matching + branch-specific context generation
│   ├── step04_generate_pairs.py     # Symmetric contrastive pair generation
│   ├── step05_finetune.py           # BGE-M3 / Contriever fine-tuning (calls models/)
│   ├── step06_embedding.py          # Multi-method vector embedding generation
│   ├── step07_evaluate.py           # Recall@10 & NDCG@10 benchmarking
│   ├── oracle_eval.py               # Pipeline sanity check using raw DTW price signals
│   └── price_concat_eval.py         # Experiment: news + price vector concatenation
├── scripts/
│   ├── data_download.py             # Download OHLCV prices, detect anomalies, download news
│   └── download_index.py            # Helper to download/cache Faiss index assets
├── .env.example                     # API key template (copy to .env, never committed)
├── docs/
│   ├── smoke_run.md                 # Step-by-step 3-epoch smoke-run guide & pass criteria
│   ├── known_risks.md               # Ranked risk register (L0 / L1 / L2) + ruff findings
│   └── review_log.md                # File-by-file code review tracker
├── Datasets/                        # Local data cache (parquet, jsonl, CSV)
├── logs/                            # Per-run logs: {NewsAgg}_{EmbModel}/{run_id}/
│   ├── agg_bge/
│   ├── sum_bge/
│   └── (contriever variants, etc.)
└── checkpoints/                     # Saved fine-tuned model weights
    ├── bge_agg_finetuned/
    ├── bge_sum_finetuned/
    └── model/                       # Legacy checkpoint slot
```

## Experiment Matrix

Each pipeline run is parameterised by two axes:

| Flag | Values | Description |
|---|---|---|
| `--NewsAgg` | `agg` / `sum` / `pool` | How daily news is aggregated into 5-day windows |
| `--EmbModel` | `bge` / `contriever` | Base embedding model |

`agg` concatenates raw headline text. `sum` uses vLLM-generated market-context summaries (early fusion). `pool` mean-pools daily dense vectors without a text summary.

## Environment

```bash
export PATH="/mnt/raid1/ken/vllm_env/bin:$PATH"
cd /mnt/raid1/ken/Capstone_data/Capstone_Context
```

## Execution Flow

Run stages in order. `--NewsAgg` and `--EmbModel` override `config.experiment.*` at runtime; no need to edit the YAML by hand.

### Step 02: Preprocessing

```bash
python main.py --step 02
```

Cleans raw news/price data, splits anomaly windows, and matches news to anomaly dates. Branch-agnostic; run once.

### Step 03: Context Generation

**Compute DTW ground truth** (branch-agnostic, run once):

```bash
python main.py --step 03 --RetModel dtw
```

**Build 5-day context windows per branch:**

```bash
# AGG branch: raw headline concatenation
python main.py --step 03 --NewsAgg agg --EmbModel bge

# SUM branch: vLLM market-context summarization (GPU required)
python main.py --step 03 --NewsAgg sum --EmbModel bge

# POOL branch: daily vector pooling (vectors produced here; Step 06 is a no-op)
python main.py --step 03 --NewsAgg pool --EmbModel bge

# POOL text5d variant: encode AGG/SUM text as one vector per 5D window
python main.py --step 03 --NewsAgg pool --RetModel text --EmbModel bge
```

### Step 04: Training Data Generation

```bash
python main.py --step 04 --NewsAgg agg --EmbModel bge
python main.py --step 04 --NewsAgg sum --EmbModel bge
# Pool text5d variant:
python main.py --step 04 --NewsAgg pool --EmbModel bge --variant text5d
```

Generates symmetric contrastive pairs (`train` / `val` / `test` splits). Date boundaries are hard-coded: `val_start=2025-01-29`, `test_start=2025-04-10`.

### Step 05: Fine-tuning (GPU required)

```bash
python main.py --step 05 --NewsAgg agg  --EmbModel bge
python main.py --step 05 --NewsAgg sum  --EmbModel bge
python main.py --step 05 --NewsAgg agg  --EmbModel contriever
python main.py --step 05 --NewsAgg sum  --EmbModel contriever
python main.py --step 05 --NewsAgg pool --EmbModel bge
python main.py --step 05 --NewsAgg pool --EmbModel contriever
```

Checkpoints land in `checkpoints/{EmbModel}_{NewsAgg}_finetuned/`. Per-epoch metrics (Recall@10 / NDCG@10) are written to `logs/{NewsAgg}_{EmbModel}/{run_id}/metrics.csv` and streamed to wandb. Append `--notify` for a desktop notification on completion.

See `docs/smoke_run.md` for the recommended 3-epoch smoke-run verification before committing to a full training run.

### Step 06: Embedding Generation

```bash
# Fine-tuned embeddings (default; requires Step 05 checkpoint)
python main.py --step 06 --NewsAgg agg --EmbModel bge

# Pretrained baseline (no Step 05 checkpoint needed)
python main.py --step 06 --NewsAgg agg --EmbModel bge --RetModel base
```

POOL branch skips this step; vectors were already produced in Step 03.

### Step 07: Evaluation

```bash
# Fine-tuned dense retrieval (BGE-M3 or Contriever)
python main.py --step 07 --NewsAgg agg --EmbModel bge     --RetModel bge
python main.py --step 07 --NewsAgg sum --EmbModel contriever --RetModel contriever

# BM25 lexical baseline
python main.py --step 07 --NewsAgg agg --EmbModel bge --RetModel bm25

# Pool text5d variant
python main.py --step 07 --NewsAgg pool --EmbModel bge --variant text5d
```

Reports Recall@10 and NDCG@10 against the DTW ground truth over the evaluation window (`evaluate.eval_start_date` → `evaluate.eval_end_date` in `config.yaml`).

To evaluate with pretrained (pre-fine-tuning) embeddings, set `evaluate.dense_embedding_source: pretrained` in `config.yaml` and use `--RetModel base` in Step 06.

## Step 01: Data Download

Before running the main pipeline, raw data must be downloaded. All three sub-steps are handled by `scripts/data_download.py`.

### Setup

```bash
# Install dependencies (if not already available)
pip install yfinance requests python-dotenv

# Set your Polygon.io API key (required for news download only)
cp .env.example .env
# Edit .env and fill in: POLYGON_API_KEY=your_key_here
```

### Download everything at once

```bash
python scripts/data_download.py --step all --output_dir ./Datasets
```

### Or run individual sub-steps

```bash
# 1. Download 5-year daily OHLCV data via Yahoo Finance (free, no key needed)
python scripts/data_download.py --step prices --output_dir ./Datasets

# 2. Detect price anomalies (rolling Z-score > 2.0) per stock
python scripts/data_download.py --step anomalies --output_dir ./Datasets

# 3. Download news articles via Polygon.io (requires API key)
python scripts/data_download.py --step news --output_dir ./Datasets
```

### Output directories (created automatically)

| Directory | Contents |
|---|---|
| `Datasets/nasdaq100_prices_5yrs_yfinance/` | One `{TICKER}_daily.csv` per stock |
| `Datasets/nasdaq100_anomalies_per_stock/`  | One `{TICKER}_anomalies.csv` per stock |
| `Datasets/nasdaq100_news_full/`            | One `{TICKER}_news.csv` per stock |

### Key options

| Flag | Default | Description |
|---|---|---|
| `--output_dir` | `./Datasets` | Root folder for all outputs |
| `--start_date` | 5 years ago | Download start date (YYYY-MM-DD) |
| `--end_date` | today | Download end date (YYYY-MM-DD) |
| `--z_threshold` | `2.0` | Z-score threshold for anomaly detection |
| `--page_sleep` | `15.0` | Seconds between Polygon.io pages (free tier limit) |
| `--polygon_api_key` | from `.env` | Override key via CLI (not recommended in scripts) |

All steps support **resume / checkpoint**: if a ticker's output file already exists it is skipped automatically.

---

## Diagnostic Scripts

These scripts run outside the main pipeline and do not require `main.py`:

```bash
# Sanity-check the eval pipeline using raw DTW price signals (should yield ~1.0 Recall@10)
python pipeline/oracle_eval.py --scorer dtw

# Experiment: news + price vector concatenation (no fine-tuning required)
python pipeline/price_concat_eval.py
```

## Monitoring a Training Run

```bash
# Live heartbeat (updates every heartbeat_sec seconds)
tail -f logs/sum_bge/<run_id>/run_status.txt

# Check metrics after training
head -n 10 logs/sum_bge/<run_id>/metrics.csv

# Confirm all artifacts exist
ls logs/sum_bge/<run_id>/
# Expected: metrics.csv  loss_curves.png  retrieval_curves.png  val_metrics_curves.png  run_status.txt  wandb/
```

## Dry Run / Path Validation

```bash
python main.py --step 05 --NewsAgg sum --EmbModel bge --dry-run
```

Validates all configured paths without executing any computation.

## Known Issues

See `docs/known_risks.md` for the full ranked risk register. Key items:

- **L1** `ContrastiveTrainer` has no real `val_loss` signal; only `InformationRetrievalEvaluator` scores are available for early-stopping.
- **L1** `_load_jsonl` silently drops malformed JSON lines (`except Exception: continue`).
- **L2** `Datasets/venv/` was accidentally committed and bloats the repo.
- **L2** The second half of `config.yaml` contains a commented legacy block pending cleanup.
