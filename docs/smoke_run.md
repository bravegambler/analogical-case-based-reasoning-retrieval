# 3-Epoch Smoke Run for Step 05 Instrumentation

**Goal:** Run a 3-epoch micro-training on each of the 6 branch × model combinations first, and verify that every epoch’s `train_loss / val_loss / subset_recall@10 / subset_ndcg@10` land correctly in wandb and `logs/{branch}_{model}/{run_id}/metrics.csv`, and that `loss_curves.png` / `retrieval_curves.png` are produced at the end. After this works, increase epochs for full training.

## Prerequisites

1. In `config.yaml`, set `finetune.epochs: 3` (keep the default 3; raise before full training).
2. In `config.yaml`, set `finetune.subset_eval.enabled: true` (default is already on).
3. Activate the environment (e.g. your existing `vllm_env`):
   ```bash
   export PATH="/mnt/raid1/ken/vllm_env/bin:$PATH"
   # Optional: source /path/to/vllm.env
   cd /mnt/raid1/ken/Capstone_data/Capstone_Recode
   ```
4. Before each combination, confirm step 04 has already produced `train/val.(jsonl|parquet)` for that combo; if not, run:
   ```bash
   python main.py --step 04 --NewsAgg <b> --EmbModel <m>
   ```

## Run the 6 combinations one by one

| # | branch | model | trainer | Command |
|---|---|---|---|---|
| 1 | agg  | bge        | ContrastiveTrainer  | `python main.py --step 05 --NewsAgg agg  --EmbModel bge` |
| 2 | agg  | contriever | ContrastiveTrainer  | `python main.py --step 05 --NewsAgg agg  --EmbModel contriever` |
| 3 | sum  | bge        | ContrastiveTrainer  | `python main.py --step 05 --NewsAgg sum  --EmbModel bge` |
| 4 | sum  | contriever | ContrastiveTrainer  | `python main.py --step 05 --NewsAgg sum  --EmbModel contriever` |
| 5 | pool | bge        | VectorBasedTrainer  | `python main.py --step 05 --NewsAgg pool --EmbModel bge` |
| 6 | pool | contriever | VectorBasedTrainer  | `python main.py --step 05 --NewsAgg pool --EmbModel contriever` |

> `--NewsAgg` / `--EmbModel` override `config.experiment.NewsAgg` / `EmbModel` at runtime; you do not need to edit the yaml by hand.

## After each run, check

```bash
# Latest run directory
LATEST=$(ls -td logs/<branch>_<model>/*/ | head -1)
echo "$LATEST"
head -n 10 "$LATEST/metrics.csv"
ls "$LATEST"   # Expect metrics.csv, loss_curves.png, retrieval_curves.png
```

Expected `metrics.csv` columns: `epoch,train_loss,val_loss,subset_recall@10,subset_ndcg@10,subset_n,lr,temperature`.

## Pass criteria (all three)

1. `metrics.csv` has 3 rows (3 epochs); every column is non-empty (`temperature` appears for pool only).
2. `loss_curves.png` and `retrieval_curves.png` exist and are non-zero size.
3. The same run appears in the wandb UI with curves for `train_loss / val_loss / subset_recall@10 / subset_ndcg@10`.

## If something fails

- **`subset_n` stays 0:** `EpochEvaluator` did not find a usable subset corpus. Check `data.news_5d_summarized_dir` for files for that branch/model (`*_5d_summaries.csv` for agg/sum, `*_5d_pool.parquet` for pool).
- **Empty `val_loss`:** Check that `data.val_jsonl` (agg/sum) or `val.parquet` (pool) exists and is non-empty.
- **`EpochEvaluator disabled (init failed)`:** See `docs/known_risks.md` first; often DTW GT has no hits in the eval window—check `evaluate.eval_start_date` / `evaluate.eval_end_date` and `dtw_results_dir`.

## After passing, run full training

Set `finetune.epochs` in `config.yaml` to what you want (e.g. 10–20), then run the same six commands. Curves, CSV, and wandb schema stay the same so you can compare directly with the smoke run.
