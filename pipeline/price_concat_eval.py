"""
pipeline/price_concat_eval.py — News + Price Retrieval Experiment
==================================================================
Purpose
-------
Test whether concatenating a historical (past-N-day) price context
vector to the pretrained news embedding breaks through the ~0.0002
Recall@10 ceiling observed with news-only retrieval.

This script is a *pure retrieval* experiment:
  - No fine-tuning.
  - No GPU required (news embeddings are already computed; price is
    just a numpy vector; similarity is CPU cosine).
  - Should match ``models/Retriever.py`` + ``step07_evaluate.compute_metrics``:
    retrieved tuples use **Window_Start** (second field) to align with
    ground-truth ``Match_Start_Date``; temporal mask uses **corpus**
    ``Window_End`` < ``Query_Date``; self-exclude uses the resolved
    (ticker, Window_End) used for the query vector.

Modes
-----
--mode news
    Pure pretrained BGE news embedding (sanity baseline; should reproduce
    the ~0.0002 number from Step 07 pretrained eval).

--mode price
    Past ``--price_days`` trading days of price signal (strictly before
    the query date for queries; ``<= Window_End`` for corpus rows).
    Representation is chosen by ``--price_repr``:

    - ``zscore``: last ``N`` closes, window z-score (default; matches prior runs).
    - ``return``: ``N`` simple daily returns from ``N+1`` closes.
    - ``logreturn``: ``N`` log returns from ``N+1`` closes.
    - ``logreturn_zscore``: ``N`` log returns, then window z-score on returns.

    Use ``--price-repr-sweep`` (price mode only) to run all four in one process
    after a single load of embeddings and prices.

--mode concat
    L2-normalised blocks of news and price concatenated with weights
    sqrt(alpha) and sqrt(1 - alpha). Tests whether the two signals are
    complementary. ``--alpha`` is the news block weight.

Usage
-----
    cd /mnt/raid1/ken/Capstone_data/Capstone_Recode
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode news
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode price --price_days 30 --price_repr zscore
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode price --price_days 30 --price_repr logreturn
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode price --price_days 30 --price-repr-sweep
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode concat --price_days 30 --alpha 0.5
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode concat --price_days 30 --alpha-sweep
    /mnt/raid1/ken/vllm_env/bin/python pipeline/price_concat_eval.py --mode concat --price_days 30 --alpha-sweep 0,0.25,0.5,0.75,1

Options
-------
  --embeddings_dir  Override news-embedding parquet dir (default: pretrained BGE).
  --price_days      Past-trading-day window length for the price vector.
  --alpha           News block weight in [0,1] for concat mode (single run).
  --alpha-sweep     Comma-separated α list, or bare flag for 0,0.1,...,1 (concat only).
  --limit           Evaluate only first N GT queries (smoke test).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import get_logger, load_config, load_dtw_ground_truth, parse_date

logger = get_logger("price_concat_eval")

# Past-price vector representations (length N = --price_days for all).
PRICE_REPRS = ("zscore", "return", "logreturn", "logreturn_zscore")


# --------------------------------------------------------------------
# Metric — identical semantics to step07_evaluate.compute_metrics
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
    for rank, (_, r_date, _) in enumerate(retrieved[:top_k], 1):
        r_dt = parse_date(r_date)
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
# Price utilities
# --------------------------------------------------------------------
def _load_ticker_prices(prices_dir: str) -> Dict[str, pd.DataFrame]:
    """Return {ticker: df with Date (sorted asc) + Close columns}."""
    out: Dict[str, pd.DataFrame] = {}
    files = sorted(glob.glob(os.path.join(prices_dir, "*_daily.csv")))
    for f in files:
        ticker = os.path.basename(f).replace("_daily.csv", "")
        try:
            df = pd.read_csv(f, usecols=["Date", "Close"])
        except Exception:
            continue
        df["_dt"] = df["Date"].apply(parse_date)
        df = df.dropna(subset=["_dt", "Close"]).sort_values("_dt").reset_index(drop=True)
        df["Close"] = df["Close"].astype(np.float64)
        out[ticker] = df
    return out


def _past_price_vec(
    price_df: pd.DataFrame,
    end_dt,
    n_days: int,
    strict_before: bool,
    price_repr: str = "zscore",
) -> Optional[np.ndarray]:
    """Return a length-``n_days`` price feature vector ending at ``end_dt``.

    ``price_repr``:
      - ``zscore``: last ``n_days`` closes, z-scored within the window.
      - ``return`` / ``logreturn``: ``n_days`` returns from ``n_days+1`` closes.
      - ``logreturn_zscore``: ``n_days`` log returns, then z-scored.

    If strict_before=True, require dates strictly < end_dt (queries).
    Else <= end_dt (corpus rows at Window_End).
    """
    if price_repr not in PRICE_REPRS:
        raise ValueError(f"price_repr must be one of {PRICE_REPRS}, got {price_repr!r}")
    if price_df is None or price_df.empty:
        return None
    mask = price_df["_dt"] < end_dt if strict_before else price_df["_dt"] <= end_dt
    RetModel = price_df.loc[mask, "Close"].to_numpy()
    need = n_days + 1 if price_repr != "zscore" else n_days
    if len(RetModel) < need:
        return None
    closes = RetModel[-need:].astype(np.float64)
    if not np.all(np.isfinite(closes)):
        return None
    if price_repr != "zscore" and np.any(closes <= 0):
        return None

    if price_repr == "zscore":
        vec = closes
        mu, sd = vec.mean(), vec.std()
        if sd < 1e-12:
            return None
        return (vec - mu) / sd

    # n_days returns from n_days+1 closes
    if price_repr == "return":
        r = closes[1:] / closes[:-1] - 1.0
    else:
        r = np.log(closes[1:] / closes[:-1])
    if not np.all(np.isfinite(r)) or r.size != n_days:
        return None

    if price_repr in ("return", "logreturn"):
        return r

    # logreturn_zscore
    mu, sd = r.mean(), r.std()
    if sd < 1e-12:
        return None
    return (r - mu) / sd


# --------------------------------------------------------------------
# Corpus / query construction
# --------------------------------------------------------------------
def _load_news_embeddings(emb_dir: str) -> pd.DataFrame:
    """Load all *_5d_summaries.parquet into one long DataFrame."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*_5d_summaries.parquet")))
    if not files:
        raise RuntimeError(f"No parquets found under {emb_dir}")
    frames = []
    for f in tqdm(files, desc="Load news embeddings"):
        df = pd.read_parquet(
            f, columns=["ticker", "Window_Start", "Window_End", "embedding"]
        )
        if df.empty:
            continue
        df["Window_Start"] = df["Window_Start"].apply(parse_date)
        df["Window_End"] = df["Window_End"].apply(parse_date)
        df = df.dropna(subset=["Window_Start", "Window_End"])
        frames.append(df)
    if not frames:
        raise RuntimeError(f"All parquets empty in {emb_dir}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "Window_End"]).reset_index(drop=True)
    return out


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def _build_vector(
    news_emb: Optional[np.ndarray],
    price_vec: Optional[np.ndarray],
    mode: str,
    alpha: float,
) -> Optional[np.ndarray]:
    """Combine news and price vectors per --mode. Returns L2-normalised vec."""
    if mode == "news":
        if news_emb is None:
            return None
        return _l2(news_emb.astype(np.float64))
    if mode == "price":
        if price_vec is None:
            return None
        return _l2(price_vec.astype(np.float64))
    # concat
    if news_emb is None or price_vec is None:
        return None
    n = _l2(news_emb.astype(np.float64))
    p = _l2(price_vec.astype(np.float64))
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.concatenate([np.sqrt(a) * n, np.sqrt(1.0 - a) * p])


# --------------------------------------------------------------------
# Corpus + retrieval (split for --alpha-sweep on concat)
# --------------------------------------------------------------------
def _build_corpus_matrix(
    news_df: pd.DataFrame,
    price_map: Dict[str, pd.DataFrame],
    mode: str,
    alpha: float,
    price_days: int,
    price_repr: str = "zscore",
    desc: str = "Build corpus",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """Returns (db, end_dts, tickers_arr, start_strs, n_dropped)."""
    corpus_vecs: List[np.ndarray] = []
    corpus_meta: List[Tuple[str, str, object]] = []
    dropped = 0
    for row in tqdm(news_df.itertuples(index=False), total=len(news_df), desc=desc):
        ticker = row.ticker
        win_start = row.Window_Start
        win_end = row.Window_End
        news_emb = row.embedding if mode != "price" else None
        price_vec = None
        if mode != "news":
            price_vec = _past_price_vec(
                price_map.get(ticker),
                win_end,
                price_days,
                strict_before=False,
                price_repr=price_repr,
            )
        v = _build_vector(news_emb, price_vec, mode, alpha)
        if v is None:
            dropped += 1
            continue
        corpus_vecs.append(v)
        corpus_meta.append((ticker, win_start.strftime("%Y-%m-%d"), win_end))

    if not corpus_vecs:
        return None, None, None, None, dropped
    db = np.stack(corpus_vecs, axis=0).astype(np.float32)
    end_dts = np.array([m[2] for m in corpus_meta], dtype=object)
    tickers_arr = np.array([m[0] for m in corpus_meta])
    start_strs = np.array([m[1] for m in corpus_meta])
    return db, end_dts, tickers_arr, start_strs, dropped


def _retrieval_query_loop(
    mode: str,
    alpha: float,
    price_days: int,
    price_repr: str,
    db: np.ndarray,
    end_dts: np.ndarray,
    tickers_arr: np.ndarray,
    start_strs: np.ndarray,
    news_by_ticker: Dict[str, pd.DataFrame],
    price_map: Dict[str, pd.DataFrame],
    gt_items: List,
    top_k: int,
    proximity_days: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """
    Returns (recalls, ndcgs, skipped_no_query, skipped_no_price, wall_seconds).
    """
    recalls: List[float] = []
    ndcgs: List[float] = []
    skipped_no_query = 0
    skipped_no_price = 0
    t0 = time.time()
    desc_extra = ""
    if mode == "concat":
        desc_extra = f" α={alpha} repr={price_repr}"
    elif mode == "price":
        desc_extra = f" repr={price_repr}"
    pbar = tqdm(gt_items, desc=f"Retrieve[{mode}]{desc_extra}")

    for (q_t, q_date), gt_matches in pbar:
        q_dt = parse_date(q_date)
        if q_dt is None:
            skipped_no_query += 1
            continue

        q_news = None
        resolved_end = None
        if mode != "price":
            tdf = news_by_ticker.get(q_t)
            if tdf is not None and not tdf.empty:
                valid = tdf[tdf["Window_End"] <= q_dt]
                if not valid.empty:
                    last_row = valid.iloc[-1]
                    q_news = last_row["embedding"]
                    resolved_end = last_row["Window_End"]
            if q_news is None:
                skipped_no_query += 1
                continue

        q_price = None
        if mode != "news":
            q_price = _past_price_vec(
                price_map.get(q_t),
                q_dt,
                price_days,
                strict_before=True,
                price_repr=price_repr,
            )
            if q_price is None:
                skipped_no_price += 1
                continue

        q_vec = _build_vector(q_news, q_price, mode, alpha)
        if q_vec is None:
            skipped_no_query += 1
            continue
        q_vec = q_vec.astype(np.float32)

        mask = end_dts < q_dt
        sims = db @ q_vec
        sims = np.where(mask, sims, -np.inf)

        if mode != "price" and resolved_end is not None:
            self_mask = (tickers_arr == q_t) & (end_dts == resolved_end)
            sims[self_mask] = -np.inf

        n_keep = max(50, top_k * 5)
        order = np.argpartition(-sims, min(n_keep, len(sims) - 1))[:n_keep]
        order = order[np.argsort(-sims[order])]

        retrieved: List[Tuple[str, str, float]] = []
        for j in order:
            s = float(sims[j])
            if not np.isfinite(s):
                break
            retrieved.append((tickers_arr[j], start_strs[j], s))
            if len(retrieved) >= n_keep:
                break

        r, n = _compute_metrics(
            retrieved, gt_matches, top_k=top_k, proximity_days=proximity_days
        )
        recalls.append(r)
        ndcgs.append(n)

    dt = time.time() - t0
    r_arr = np.asarray(recalls, dtype=np.float64)
    n_arr = np.asarray(ndcgs, dtype=np.float64)
    return r_arr, n_arr, skipped_no_query, skipped_no_price, dt


def _parse_alpha_sweep(s: str) -> List[float]:
    out: List[float] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty --alpha-sweep list.")
    for a in out:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {a}.")
    return out


def _parse_price_repr_sweep(s: str) -> List[str]:
    out: List[str] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if part not in PRICE_REPRS:
            raise ValueError(
                f"Unknown price_repr {part!r}; allowed: {', '.join(PRICE_REPRS)}"
            )
        out.append(part)
    if not out:
        raise ValueError("Empty --price-repr-sweep list.")
    return out


# --------------------------------------------------------------------
# Main retrieval loop
# --------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    conf = load_config(args.config)
    data = conf["data"]
    ev = conf["evaluate"]

    start_dt = parse_date(ev["eval_start_date"])
    end_dt = parse_date(ev["eval_end_date"])
    top_k = int(ev.get("top_k", 10))
    proximity_days = int(ev.get("proximity_days", 5))

    emb_dir = args.embeddings_dir or os.path.join(
        data["dataset_dir"], "nasdaq100_embeddings_agg_bge_pretrained"
    )
    prices_dir = data["prices_dir"]

    sweep_alphas: Optional[List[float]] = None
    if getattr(args, "alpha_sweep", None):
        if args.mode != "concat":
            raise SystemExit(
                "Use --mode concat with --alpha-sweep (concat-only)."
            )
        sweep_alphas = _parse_alpha_sweep(args.alpha_sweep)

    sweep_reprs: Optional[List[str]] = None
    if getattr(args, "price_repr_sweep", None):
        if args.mode != "price":
            raise SystemExit(
                "Use --mode price with --price-repr-sweep (price-only)."
            )
        sweep_reprs = _parse_price_repr_sweep(args.price_repr_sweep)

    if sweep_alphas is not None and sweep_reprs is not None:
        raise SystemExit("Use only one of --alpha-sweep or --price-repr-sweep.")

    logger.info(
        f"Mode={args.mode} price_days={args.price_days} "
        f"price_repr={args.price_repr} alpha={args.alpha}"
    )
    logger.info(f"Eval window: {ev['eval_start_date']} → {ev['eval_end_date']}")
    logger.info(f"top_k={top_k} proximity_days={proximity_days}")
    logger.info(f"News emb dir: {emb_dir}")
    logger.info(f"Prices dir:  {prices_dir}")
    if sweep_alphas is not None:
        logger.info(f"Alpha sweep: {sweep_alphas} (N={args.price_days})")
    if sweep_reprs is not None:
        logger.info(f"Price-repr sweep: {sweep_reprs} (N={args.price_days})")

    # --- Load news + prices once
    news_df = _load_news_embeddings(emb_dir)
    logger.info(
        f"Loaded {len(news_df)} news-embedding rows across "
        f"{news_df['ticker'].nunique()} tickers"
    )
    price_map = _load_ticker_prices(prices_dir)
    logger.info(f"Loaded price series for {len(price_map)} tickers")

    news_by_ticker: Dict[str, pd.DataFrame] = {
        t: g.sort_values("Window_End").reset_index(drop=True)
        for t, g in news_df.groupby("ticker")
    }

    gt = load_dtw_ground_truth(data["dtw_results_dir"], start_dt, end_dt)
    logger.info(f"GT queries in eval window: {len(gt)}")

    gt_items = list(gt.items())
    if args.limit and args.limit > 0:
        gt_items = gt_items[: args.limit]
        logger.info(f"(--limit) evaluating first {len(gt_items)} queries only")

    # --- alpha sweep: rebuild corpus per alpha, reuse GT / news
    if sweep_alphas is not None:
        summary_rows: List[Tuple[float, float, float]] = []
        t_scan = time.time()
        for a in sweep_alphas:
            db, end_dts, tickers_arr, start_strs, dropped = _build_corpus_matrix(
                news_df,
                price_map,
                "concat",
                a,
                args.price_days,
                price_repr=args.price_repr,
                desc=f"Build corpus (α={a})",
            )
            if db is None:
                logger.error("Empty corpus (concat).")
                return
            logger.info(
                f"Corpus α={a}: {db.shape[0]} vec | dim={db.shape[1]} | "
                f"dropped={dropped}"
            )
            r_arr, n_arr, s_q, s_p, t_loop = _retrieval_query_loop(
                "concat",
                a,
                args.price_days,
                args.price_repr,
                db,
                end_dts,
                tickers_arr,
                start_strs,
                news_by_ticker,
                price_map,
                gt_items,
                top_k,
                proximity_days,
            )
            mean_r, mean_n = r_arr.mean(), n_arr.mean()
            summary_rows.append((a, mean_r, mean_n))
            logger.info(
                f"  α={a:.4f}  Recall@{top_k}={mean_r:.6f}  NDCG@{top_k}={mean_n:.6f}  "
                f"eval={len(r_arr)}  skip_q={s_q}  skip_p={s_p}  t={t_loop:.1f}s"
            )

        w_alpha, w_r, w_n = 8, 14, 14
        logger.info("--- alpha sweep (concat) — summary ---")
        logger.info(
            f"{'alpha':>{w_alpha}}  {f'R@{top_k}':>{w_r}}  {f'NDCG@{top_k}':>{w_n}}"
        )
        logger.info("-" * (w_alpha + w_r + w_n + 4))
        for a, mean_r, mean_n in summary_rows:
            logger.info(
                f"{a:>{w_alpha}.4f}  {mean_r:>{w_r}.6f}  {mean_n:>{w_n}.6f}"
            )
        logger.info("Total sweep time: %.1fs", time.time() - t_scan)
        return

    # --- price-repr sweep: one load, multiple corpus builds (price mode)
    if sweep_reprs is not None:
        summary_rows: List[Tuple[str, float, float]] = []
        t_scan = time.time()
        for pr in sweep_reprs:
            db, end_dts, tickers_arr, start_strs, dropped = _build_corpus_matrix(
                news_df,
                price_map,
                "price",
                args.alpha,
                args.price_days,
                price_repr=pr,
                desc=f"Build corpus (repr={pr})",
            )
            if db is None:
                logger.error("Empty corpus (price).")
                return
            logger.info(
                f"Corpus repr={pr}: {db.shape[0]} vec | dim={db.shape[1]} | "
                f"dropped={dropped}"
            )
            r_arr, n_arr, s_q, s_p, t_loop = _retrieval_query_loop(
                "price",
                args.alpha,
                args.price_days,
                pr,
                db,
                end_dts,
                tickers_arr,
                start_strs,
                news_by_ticker,
                price_map,
                gt_items,
                top_k,
                proximity_days,
            )
            mean_r, mean_n = r_arr.mean(), n_arr.mean()
            summary_rows.append((pr, mean_r, mean_n))
            logger.info(
                f"  repr={pr:<18}  Recall@{top_k}={mean_r:.6f}  NDCG@{top_k}={mean_n:.6f}  "
                f"eval={len(r_arr)}  skip_q={s_q}  skip_p={s_p}  t={t_loop:.1f}s"
            )

        w_pr, w_r, w_n = 20, 14, 14
        logger.info("--- price_repr sweep (price) — summary ---")
        logger.info(
            f"{'repr':>{w_pr}}  {f'R@{top_k}':>{w_r}}  {f'NDCG@{top_k}':>{w_n}}"
        )
        logger.info("-" * (w_pr + w_r + w_n + 4))
        for pr, mean_r, mean_n in summary_rows:
            logger.info(f"{pr:>{w_pr}}  {mean_r:>{w_r}.6f}  {mean_n:>{w_n}.6f}")
        logger.info("Total sweep time: %.1fs", time.time() - t_scan)
        return

    # --- single run (news / price / one concat)
    db, end_dts, tickers_arr, start_strs, dropped = _build_corpus_matrix(
        news_df,
        price_map,
        args.mode,
        args.alpha,
        args.price_days,
        price_repr=args.price_repr,
        desc="Build corpus",
    )
    if db is None:
        logger.error("Empty corpus after vector construction. Check inputs.")
        return
    dim = db.shape[1]
    logger.info(
        f"Corpus ready: {db.shape[0]} vectors | dim={dim} | dropped={dropped}"
    )

    r_arr, n_arr, skipped_no_query, skipped_no_price, dt = _retrieval_query_loop(
        args.mode,
        args.alpha,
        args.price_days,
        args.price_repr,
        db,
        end_dts,
        tickers_arr,
        start_strs,
        news_by_ticker,
        price_map,
        gt_items,
        top_k,
        proximity_days,
    )

    if r_arr.size == 0:
        logger.warning("No queries evaluated.")
        return

    logger.info(
        f"Evaluated {len(r_arr)} / {len(gt_items)} queries in {dt:.1f}s "
        f"(skipped_no_query={skipped_no_query}, skipped_no_price={skipped_no_price})"
    )
    logger.info(
        f"Per-query: recall nonzero={int((r_arr > 0).sum())}/{len(r_arr)} "
        f"min={r_arr.min():.6f} max={r_arr.max():.6f} "
        f"median={np.median(r_arr):.6f}"
    )
    logger.info(
        f"Per-query: ndcg   nonzero={int((n_arr > 0).sum())}/{len(r_arr)} "
        f"min={n_arr.min():.6f} max={n_arr.max():.6f} "
        f"median={np.median(n_arr):.6f}"
    )
    tag = args.mode.upper()
    if args.mode == "concat":
        tag += f"(alpha={args.alpha})"
    if args.mode in ("price", "concat"):
        tag += f"[N={args.price_days},repr={args.price_repr}]"
    logger.info(
        f"[{tag}] Recall@{top_k}: {r_arr.mean():.6f} | NDCG@{top_k}: {n_arr.mean():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["news", "price", "concat"]
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default=None,
        help="Override news-embedding parquet dir. "
             "Default: ${dataset_dir}/nasdaq100_embeddings_agg_bge_pretrained"
    )
    parser.add_argument(
        "--price_days", type=int, default=30,
        help="Past-trading-day window length for the price vector.",
    )
    parser.add_argument(
        "--price_repr",
        type=str,
        default="zscore",
        choices=list(PRICE_REPRS),
        help="How to build the length-N price block from daily Close "
        "(zscore closes vs returns from N+1 closes; concat uses the same).",
    )
    parser.add_argument(
        "--price-repr-sweep",
        type=str,
        nargs="?",
        const=",".join(PRICE_REPRS),
        default=None,
        metavar="REPRS",
        help="Price mode only: comma-separated repr list, or bare flag for "
        f"{','.join(PRICE_REPRS)}. Single load; prints summary table.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="News-block weight in [0,1] for concat mode (ignored if --alpha-sweep is set).",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        nargs="?",
        const="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        default=None,
        metavar="ALPHAS",
        help="Concat only: comma-separated α in [0,1], or use flag alone for default "
        "0,0.1,...,1. Rebuilds corpus per α; long run.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If > 0, evaluate only first N GT queries (smoke test).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
