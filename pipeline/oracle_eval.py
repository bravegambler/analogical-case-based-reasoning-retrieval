"""
pipeline/oracle_eval.py — Evaluation Pipeline Sanity Check
===========================================================
Purpose
-------
Before blaming the model for near-zero Recall@10 / NDCG@10, rule out bugs
in the evaluation pipeline itself (date handling, temporal masking,
`compute_metrics` proximity logic, GT/corpus key alignment).

This script replaces the learned news embedding with the *exact* signal the
DTW ground truth is built from — the Z-normalised post-anomaly price path
(from `data.post_anomaly_ts_dir`) — and runs a retrieval evaluation that
mirrors `pipeline/step07_evaluate.compute_metrics` semantically.

Scorers
-------
--scorer dtw
    Re-rank candidates by DTW distance (dtaidistance), which is how the GT
    itself is generated. Recall@10 / NDCG@10 should be close to 1.0.
    If it is not, something in the evaluation plumbing is wrong and the
    downstream model numbers cannot be trusted.

--scorer cos
    Cosine similarity on padded Z-norm vectors. A strictly looser upper
    bound (no warping) that also sanity-checks the "vector retrieval"
    path used by the real evaluators.

Usage
-----
    cd /mnt/raid1/ken/Capstone_data/Capstone_Recode
    python pipeline/oracle_eval.py --scorer dtw
    python pipeline/oracle_eval.py --scorer cos --max-len 10
    python pipeline/oracle_eval.py --scorer dtw --limit 50   # quick smoke
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import get_logger, load_config, load_dtw_ground_truth, parse_date

logger = get_logger("oracle_eval")


# --------------------------------------------------------------------
# Metric: identical semantics to step07_evaluate.compute_metrics
# (inlined to keep this script free of torch / retriever deps)
# --------------------------------------------------------------------
def _compute_metrics(
    retrieved: List[Tuple[str, str, float]],
    gt_matches: List[Tuple[str, str]],
    top_k: int = 10,
    proximity_days: int = 5,
) -> Tuple[float, float]:
    gt_list = gt_matches[:top_k]
    gt_parsed = [(t, parse_date(d)) for t, d in gt_list if parse_date(d)]

    hits = 0
    dcg = 0.0
    matched = set()
    for rank, (_, r_start, _) in enumerate(retrieved[:top_k], 1):
        r_dt = parse_date(r_start)
        if not r_dt:
            continue
        for gt_idx, (_, gt_dt) in enumerate(gt_parsed):
            if abs((r_dt - gt_dt).days) <= proximity_days:
                if gt_idx not in matched:
                    matched.add(gt_idx)
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 1)
                    break

    recall = hits / len(gt_list) if gt_list else 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_list), top_k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
def _znorm(prices: np.ndarray) -> np.ndarray:
    """Match step03._compute_dtw exactly: (prices / prices[0] - 1) * 100."""
    base = float(prices[0])
    if base == 0.0 or not np.isfinite(base):
        return prices.astype(np.double)
    return ((prices / base - 1.0) * 100.0).astype(np.double)


def _load_post_move_sequences(ts_dir: str) -> Dict[Tuple[str, str], np.ndarray]:
    """Build {(ticker, 'YYYY-MM-DD'): z_norm_vec} from *_post_moves.csv."""
    out: Dict[Tuple[str, str], np.ndarray] = {}
    files = sorted(glob.glob(os.path.join(ts_dir, "*_post_moves.csv")))
    if not files:
        return out
    for f in tqdm(files, desc="Load post-move sequences"):
        ticker = os.path.basename(f).replace("_post_moves.csv", "")
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "Close" not in df.columns or "anomaly_date" not in df.columns:
            continue
        for anom_date, group in df.groupby("anomaly_date"):
            prices = group["Close"].to_numpy(dtype=np.double)
            if len(prices) < 5:
                continue
            dt = parse_date(anom_date)
            if dt is None:
                continue
            key = (ticker, dt.strftime("%Y-%m-%d"))
            if key in out:
                continue
            out[key] = _znorm(prices)
    return out


def _pad_or_trim(vec: np.ndarray, length: int) -> np.ndarray:
    if len(vec) >= length:
        return vec[:length].astype(np.float64)
    pad = np.full(length - len(vec), vec[-1] if len(vec) > 0 else 0.0, dtype=np.float64)
    return np.concatenate([vec.astype(np.float64), pad])


# --------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------
def _score_dtw_row(q_vec: np.ndarray, corpus_vecs: List[np.ndarray]) -> np.ndarray:
    from dtaidistance import dtw as dtw_distance

    out = np.empty(len(corpus_vecs), dtype=np.float64)
    for i, c in enumerate(corpus_vecs):
        try:
            out[i] = -float(dtw_distance.distance(q_vec, c))
        except Exception:
            out[i] = -np.inf
    return out


def _score_cos_all(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    q_n = q / (np.linalg.norm(q) + 1e-12)
    return db @ q_n


# --------------------------------------------------------------------
# Main oracle loop
# --------------------------------------------------------------------
def run_oracle(conf: Dict, scorer: str, max_len: int, limit: int = 0) -> None:
    data = conf["data"]
    ev = conf["evaluate"]

    start_dt = parse_date(ev["eval_start_date"])
    end_dt = parse_date(ev["eval_end_date"])
    top_k = int(ev.get("top_k", 10))
    proximity_days = int(ev.get("proximity_days", 5))

    logger.info(f"Eval window: {ev['eval_start_date']} → {ev['eval_end_date']}")
    logger.info(f"top_k={top_k} proximity_days={proximity_days} scorer={scorer}")

    logger.info(f"Loading post-move sequences from {data['post_anomaly_ts_dir']}")
    seqs = _load_post_move_sequences(data["post_anomaly_ts_dir"])
    logger.info(f"  Loaded {len(seqs)} (ticker, anomaly_date) sequences")
    if not seqs:
        logger.error("No post-move sequences found. Check data.post_anomaly_ts_dir.")
        return

    keys = sorted(seqs.keys())
    meta = [(t, d, d) for (t, d) in keys]
    end_dts = np.array([parse_date(m[2]) for m in meta], dtype=object)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    db_mat = None
    corpus_vecs = None
    if scorer == "cos":
        db_mat = np.stack([_pad_or_trim(seqs[k], max_len) for k in keys], axis=0)
        norms = np.linalg.norm(db_mat, axis=1, keepdims=True) + 1e-12
        db_mat = db_mat / norms
        logger.info(f"  Built cosine db matrix: shape={db_mat.shape}")
    else:
        corpus_vecs = [seqs[k] for k in keys]

    gt = load_dtw_ground_truth(data["dtw_results_dir"], start_dt, end_dt)
    logger.info(f"GT queries in eval window: {len(gt)}")

    recalls: List[float] = []
    ndcgs: List[float] = []
    missing_query = 0
    evaluated = 0
    t0 = time.time()

    gt_items = list(gt.items())
    if limit and limit > 0:
        gt_items = gt_items[:limit]
        logger.info(f"  (--limit) evaluating first {len(gt_items)} queries only")

    for (q_t, q_date), gt_matches in tqdm(gt_items, desc=f"Oracle[{scorer}]"):
        if (q_t, q_date) not in key_to_idx:
            missing_query += 1
            continue
        q_idx = key_to_idx[(q_t, q_date)]
        q_dt = parse_date(q_date)
        if q_dt is None:
            missing_query += 1
            continue

        if scorer == "dtw":
            scores = _score_dtw_row(corpus_vecs[q_idx], corpus_vecs)
        else:
            scores = _score_cos_all(db_mat[q_idx], db_mat)

        mask = np.array([d is not None and d < q_dt for d in end_dts])
        scores = np.where(mask, scores, -np.inf)
        scores[q_idx] = -np.inf

        order = np.argsort(-scores)[: max(50, top_k * 5)]
        retrieved: List[Tuple[str, str, float]] = []
        for j in order:
            s = float(scores[j])
            if not np.isfinite(s):
                break
            m = meta[j]
            retrieved.append((m[0], m[1], s))
            if len(retrieved) >= max(50, top_k * 5):
                break

        r, n = _compute_metrics(retrieved, gt_matches, top_k=top_k, proximity_days=proximity_days)
        recalls.append(r)
        ndcgs.append(n)
        evaluated += 1

    dt = time.time() - t0
    if not recalls:
        logger.warning(
            "No queries evaluated. Possible causes: eval window has no GT, "
            "or post-move sequences don't cover the GT date range."
        )
        return

    r_arr = np.asarray(recalls, dtype=np.float64)
    n_arr = np.asarray(ndcgs, dtype=np.float64)
    logger.info(
        f"Evaluated {evaluated} / {len(gt_items)} queries in {dt:.1f}s "
        f"(missing_query={missing_query})"
    )
    logger.info(
        f"Per-query: recall nonzero={int((r_arr > 0).sum())}/{len(r_arr)} "
        f"min={r_arr.min():.4f} max={r_arr.max():.4f} "
        f"median={np.median(r_arr):.4f}"
    )
    logger.info(
        f"Per-query: ndcg   nonzero={int((n_arr > 0).sum())}/{len(n_arr)} "
        f"min={n_arr.min():.4f} max={n_arr.max():.4f} "
        f"median={np.median(n_arr):.4f}"
    )
    logger.info(
        f"[ORACLE {scorer.upper()}] Recall@{top_k}: {r_arr.mean():.4f} | "
        f"NDCG@{top_k}: {n_arr.mean():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Oracle sanity-check for retrieval evaluation.")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument(
        "--scorer",
        type=str,
        default="dtw",
        choices=["dtw", "cos"],
        help="dtw: should give Recall@10 near 1.0. cos: looser upper bound.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=10,
        help="Pad/trim length for cosine mode (ignored for dtw).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only evaluate first N GT queries (smoke test).",
    )
    args = parser.parse_args()

    conf = load_config(args.config)
    run_oracle(conf, scorer=args.scorer, max_len=args.max_len, limit=args.limit)


if __name__ == "__main__":
    main()
