# Capstone_Recode — Code Review Log

Go through one file at a time (one row per file) and update status when done. Suggested order: **Shared → Models → Pipeline**.

**Status**: `todo` | `reading` | `reviewed` | `risk` | `fix`
**Severity**: `L0` (blocks conclusions / leakage / invalid eval) | `L1` (affects results or reproducibility) | `L2` (tech debt, does not affect conclusions)

---

## Shared

| File | Status | Focus | Notes |
|---|---|---|---|
| `main.py` | todo | Dispatch, `NewsAgg/model` override, CUDA setup | |
| `utility.py` | todo | `parse_date`, `load_dtw_ground_truth`, `_resolve_vars`, `build_corpus_matrix` | |
| `config.yaml` | todo | Path templates, whether `${}` resolves everywhere, branch-isolated dirs / ckpt names | |
| `README.md` | todo | Matches actual commands | |
| `Overview.md` | todo | Matches actual layout | |

## Models

| File | Status | Focus | Notes |
|---|---|---|---|
| `models/Retriever.py` | todo | `_resolve_query_key`, corpus build, temporal mask | snap logic already fixed |
| `models/ContrastiveTrainer.py` | todo | opaque `model.fit`, missing loss plumbing, `_load_jsonl` swallowing errors | |
| `models/VectorBasedTrainer.py` | todo | no `val_loader`, only `train_loss`; late-init dim edge cases | |
| `models/Poolers.py` | todo | whether `AttentionPooler` actually receives mask for variable-length batches | |
| `models/train_metrics.py` | new | New in this round: `EpochEvaluator` + `MetricsWriter` | in progress |

## Pipeline

| File | Status | Focus | Notes |
|---|---|---|---|
| `pipeline/step02_preprocess.py` | todo | date formats, ticker consistency, error handling | |
| `pipeline/step03_dtw_summarize.py` | todo | DTW temporal causality, whether `Query_Date` is always a trading day, window build | `Window_End` only from `tds[4:]` |
| `pipeline/step04_generate_pairs.py` | todo | train/val/test date splits (`val_start=2025-01-29`, `test_start=2025-04-10`) | val split exists |
| `pipeline/step05_finetune.py` | todo | dispatch only | |
| `pipeline/step06_embedding.py` | todo | full-corpus encoding consistency, `Window_End` format | |
| `pipeline/step07_evaluate.py` | todo | `compute_metrics` definition, dense stats, BM25 consistency | snap stats wired in |

---

## Checklist (one-line summary per file after reading)

- **Leakage / temporal causality**: `c_date < q_date`, `Window_End < Query_Date`, val/test split boundaries
- **Key alignment**: `(ticker, Window_End)` vs `(Query_Ticker, Query_Date)` format and parsing agree
- **Column name drift**: `Ticker` vs `ticker`, `Query_Date` vs `q_date` (past footguns)
- **Swallowed exceptions**: `except Exception: continue` / `except: pass`
- **Path resolution**: `${}` expansion, `dry-run` branches
- **Device / memory**: `.to(device)`, `torch.no_grad()`, batch shapes
- **Random seeds**: `torch` / `numpy` / `random`
- **Branch isolation**: outputs do not overwrite each other when switching `NewsAgg` / `model`

---

## Example review notes

```
2026-04-21 models/Retriever.py → reviewed
  L2: parquet read errors not caught specifically; may hide low-level IO failures
  L1: db_end_dates as np.array(datetime); `<` on large corpora has some overhead
```
