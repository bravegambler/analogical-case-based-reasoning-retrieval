"""
Step 02 — Data Preprocessing
=============================
Consolidates five RetModel-steps from Capstone_data2/02_Data_Preprocessing:
  split      : Aggregate per-stock anomalies → train/val/test split (8:1:1)
  dedup      : Fuzzy-deduplicate news (anomaly + full)
  match_news : Match news in the N calendar days *before* each anomaly (excludes the anomaly day)
  timeseries : Extract post-anomaly price sequences (T+1 … T+10)

Usage:
  python main.py --step 02                  # run all RetModel-steps
  python main.py --step 02 --RetModel split      # run one RetModel-step
"""

import os
import glob
import json
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

from utility import get_logger, ensure_dir, parse_date, fuzzy_dedup

logger = get_logger("step02")


# ================================================================
# Sub-step: split
# ================================================================
def _split_anomalies(conf: Dict):
    """Aggregate per-stock anomaly CSVs and split into train/val/test."""
    data = conf["data"]
    pp = conf["preprocess"]

    input_dir = data["anomalies_per_stock_dir"]
    output_dir = ensure_dir(data["anomalies_split_dir"])

    csv_files = glob.glob(os.path.join(input_dir, "*_anomalies.csv"))
    logger.info(f"Reading anomaly files from {input_dir} ({len(csv_files)} stocks)")

    all_anomalies = []
    for fpath in csv_files:
        ticker = os.path.basename(fpath).replace("_anomalies.csv", "")
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            for _, row in df.iterrows():
                all_anomalies.append({
                    "Company": ticker,
                    "Date": str(row[date_col]),
                    "Close": row.get("Close"),
                    "Return": row.get("Return"),
                    "Z_Score": row.get("Z_Score"),
                })
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")

    if not all_anomalies:
        logger.error("No anomalies found. Check input directory.")
        return

    # Deterministic chronological split
    df_all = pd.DataFrame(all_anomalies)
    df_all["_dt"] = df_all["Date"].apply(parse_date)
    df_all = df_all.dropna(subset=["_dt"]).sort_values("_dt").drop(columns=["_dt"])

    n = len(df_all)
    train_end = int(n * pp["train_ratio"])
    val_end = int(n * (pp["train_ratio"] + pp["val_ratio"]))

    splits = {
        "train": df_all.iloc[:train_end],
        "val": df_all.iloc[train_end:val_end],
        "test": df_all.iloc[val_end:],
    }

    for name, split_df in splits.items():
        out_path = os.path.join(output_dir, f"anomalies_{name}.csv")
        split_df.to_csv(out_path, index=False)
        logger.info(f"  {name}: {len(split_df)} events → {out_path}")

    logger.info(f"Total: {n} anomalies split as {train_end}/{val_end - train_end}/{n - val_end}")


# ================================================================
# Sub-step: dedup
# ================================================================
def _dedup_news(conf: Dict):
    """Fuzzy-deduplicate raw news CSVs in-place."""
    data = conf["data"]
    pp = conf["preprocess"]
    threshold = pp["similarity_threshold"]

    news_dir = data["news_full_dir"]
    csv_files = glob.glob(os.path.join(news_dir, "*.csv"))
    logger.info(f"Deduplicating {len(csv_files)} news files (threshold={threshold})")

    total_removed = 0
    for fpath in tqdm(csv_files, desc="Dedup"):
        try:
            df = pd.read_csv(fpath)
            if df.empty or "title" not in df.columns:
                continue

            texts = df["title"].fillna("").tolist()
            keep_idx = fuzzy_dedup(texts, threshold=threshold)
            removed = len(df) - len(keep_idx)

            if removed > 0:
                df_clean = df.iloc[keep_idx]
                df_clean.to_csv(fpath, index=False)
                total_removed += removed
        except Exception as e:
            logger.warning(f"Error deduplicating {fpath}: {e}")

    logger.info(f"Deduplication complete. Removed {total_removed} duplicates.")


# ================================================================
# Sub-step: match_news
# ================================================================
def _match_anomaly_news(conf: Dict):
    """For each anomaly event, extract news from [anom_date - N, anom_date), i.e. excluding the anomaly day."""
    data = conf["data"]
    pp = conf["preprocess"]

    anomaly_dir = data["anomalies_per_stock_dir"]
    news_dir = data["news_full_dir"]
    output_dir = ensure_dir(data["anomaly_contexts_dir"])
    window_days = pp["anomaly_news_window_days"]

    anomaly_files = [f for f in os.listdir(anomaly_dir) if f.endswith(".csv")]
    logger.info(
        f"Matching news for {len(anomaly_files)} stocks (last {window_days} calendar days before each anomaly, excluding anomaly day)"
    )

    for af in tqdm(anomaly_files, desc="Match news"):
        ticker = af.replace("_anomalies.csv", "")
        anomaly_path = os.path.join(anomaly_dir, af)
        news_path = os.path.join(news_dir, f"{ticker}_news.csv")

        if not os.path.exists(news_path):
            continue

        try:
            df_anomaly = pd.read_csv(anomaly_path)
            df_anomaly["Date"] = pd.to_datetime(df_anomaly["Date"]).dt.tz_localize(None).dt.normalize()

            df_news = pd.read_csv(news_path)

            # Auto-detect date column
            date_col = None
            for col in ["published_utc", "timestamp", "datetime", "date", "time"]:
                if col in df_news.columns:
                    date_col = col
                    break
            if not date_col:
                continue

            df_news[date_col] = pd.to_datetime(df_news[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
            df_news = df_news.dropna(subset=[date_col])

            context_rows = []
            for _, anom in df_anomaly.iterrows():
                anom_date = anom["Date"]
                start = anom_date - timedelta(days=window_days)
                # Strictly before anomaly day (no news dated on the anomaly day itself).
                mask = (df_news[date_col] >= start) & (df_news[date_col] < anom_date)
                matched = df_news[mask].copy()
                if not matched.empty:
                    matched["anomaly_date"] = anom_date.strftime("%Y-%m-%d")
                    matched["z_score"] = anom.get("Z_Score")
                    context_rows.append(matched)

            if context_rows:
                result = pd.concat(context_rows, ignore_index=True)
                result.to_csv(os.path.join(output_dir, f"{ticker}_context.csv"), index=False)

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")

    logger.info("Anomaly-news matching complete.")


# ================================================================
# Sub-step: timeseries
# ================================================================
def _extract_post_anomaly_ts(conf: Dict):
    """Extract T+1 to T+10 price data after each anomaly event."""
    data = conf["data"]
    track_days = 10

    anomaly_dir = data["anomalies_per_stock_dir"]
    price_dir = data["prices_dir"]
    output_dir = ensure_dir(data["post_anomaly_ts_dir"])

    anomaly_files = [f for f in os.listdir(anomaly_dir) if f.endswith(".csv")]
    logger.info(f"Extracting post-anomaly time series for {len(anomaly_files)} stocks")

    for af in tqdm(anomaly_files, desc="Post-anomaly TS"):
        ticker = af.replace("_anomalies.csv", "")
        anomaly_path = os.path.join(anomaly_dir, af)
        price_path = os.path.join(price_dir, f"{ticker}_daily.csv")

        if not os.path.exists(price_path):
            continue

        try:
            df_anomaly = pd.read_csv(anomaly_path)
            df_price = pd.read_csv(price_path)

            df_anomaly["Date"] = pd.to_datetime(df_anomaly["Date"]).dt.tz_localize(None).dt.normalize()
            df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.tz_localize(None).dt.normalize()
            df_price = df_price.sort_values("Date").reset_index(drop=True)

            results = []
            for _, anom in df_anomaly.iterrows():
                anom_date = anom["Date"]
                loc = df_price[df_price["Date"] == anom_date].index
                if len(loc) == 0:
                    continue
                start_idx = loc[0] + 1
                end_idx = min(start_idx + track_days, len(df_price))
                segment = df_price.iloc[start_idx:end_idx].copy()
                segment["anomaly_date"] = anom_date.strftime("%Y-%m-%d")
                segment["z_score"] = anom.get("Z_Score")
                results.append(segment)

            if results:
                pd.concat(results).to_csv(
                    os.path.join(output_dir, f"{ticker}_post_moves.csv"), index=False
                )
        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")

    logger.info("Post-anomaly time series extraction complete.")


# ================================================================
# Runner
# ================================================================
_SUBSTEPS = {
    "split": _split_anomalies,
    "dedup": _dedup_news,
    "match_news": _match_anomaly_news,
    "timeseries": _extract_post_anomaly_ts,
}


def run(conf: Dict, RetModel: Optional[str] = None):
    """Execute step 02 (all RetModel-steps or a specific one)."""
    if RetModel:
        if RetModel not in _SUBSTEPS:
            raise ValueError(f"Unknown RetModel-step '{RetModel}'. Choose from: {list(_SUBSTEPS.keys())}")
        logger.info(f"Running step02/{RetModel}")
        _SUBSTEPS[RetModel](conf)
    else:
        logger.info("Running all step02 RetModel-steps sequentially")
        for name, fn in _SUBSTEPS.items():
            logger.info(f"--- {name} ---")
            fn(conf)
