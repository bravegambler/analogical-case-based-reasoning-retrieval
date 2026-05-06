# Capstone_Recode — Known Risks

L0 blocks conclusions; L1 affects results; L2 is tech debt. Add new findings from review here.

---

## L0 — Blocks conclusions

(None yet)

## L1 — Affects results / reproducibility

- **`ContrastiveTrainer.train` has no real `val_loss` signal:** only `InformationRetrievalEvaluator` scores; you cannot plot a proper train/val loss curve, and early stopping is weak.
- **`VectorBasedTrainer.train` has no val set:** only `train_loss`; hard to judge overfitting strictly.
- **`_load_jsonl` swallows errors (`except Exception: continue`):** bad JSON lines vanish quietly and training loses rows. Prefer fail-fast or warn + count.
- **Random seeds not set in one place:** `torch.manual_seed` / `numpy.random.seed` not unified at trainer entry; runs will differ run-to-run.
- **Step 04 val/test split is hard-coded on `Query_Date < val_start / test_start`:** changing the experiment window requires editing string constants in `step04_generate_pairs.py`.

## L2 — Tech debt

- The second half of `config.yaml` is a commented legacy block; move to `docs/config_history.md` or delete.
- `Datasets/venv/` was committed by mistake (local venv); it bloats the repo.
- `AttentionPooler.forward` can crash if batch seq lengths differ and the DataLoader does not pass a mask; only used when `pooling_type=="attention"`, but nothing enforces passing a mask.

---

## ruff / pyright / vulture output

### ruff (first run 2026-04-21, 67 findings)

By severity:

- **L1 — may hide errors**
  - `pipeline/step07_evaluate.py:286` — bare `except: continue` (E722); can hide non-domain exceptions
  - `pipeline/step04_generate_pairs.py:51` — `m_map` defined but unused (F841); also re-reads `m_news_csv` parquet every row (perf debt + original design not followed; should cache from outside)

- **L2 — tech debt**
  - Many `E701` (`if cond: continue` on one line); readability only
  - `pipeline/step02_preprocess.py:17` / `step03_dtw_summarize.py:11-14,20` / `step06_embedding.py:14,19` / `step07_evaluate.py:24` — unused imports (F401)

Quick auto-fix:

```bash
cd /mnt/raid1/ken/Capstone_data/Capstone_Recode
/mnt/raid1/ken/vllm_env/bin/python -m ruff check models pipeline utility.py main.py --fix
```

(Only 10 F401 auto-fixed; E701/E722 need manual line breaks / narrower exception types.)

### pyright / vulture

Not run yet—needs extra install; follow up later. Record in `review_log.md` as needed.
