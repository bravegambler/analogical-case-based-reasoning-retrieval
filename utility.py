"""
utility.py — Shared Utilities for Capstone_Recode
==================================================
Common functions used across multiple pipeline stages:
  - Config loading with variable interpolation
  - Date parsing (multi-format)
  - Data I/O helpers
  - Text deduplication
"""

import os
import re
import yaml
import glob
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# ================================================================
# Logging
# ================================================================
def get_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    """Create a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def notify_run_finished(title: str, message: str, *, sound: bool = True, desktop: bool = True) -> None:
    """
    Best-effort completion alert for long runs (Linux desktop / Cursor terminal).
    Tries: paplay on common freedesktop sounds, then canberra-gtk-play, then terminal bell.
    """
    if desktop and shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "-a", title, title, message],
            check=False,
            capture_output=True,
            timeout=5,
        )
    if not sound:
        return
    sound_paths = (
        "/usr/share/sounds/freedesktop/stereo/complete.oga",
        "/usr/share/sounds/freedesktop/stereo/bell.oga",
        "/usr/share/sounds/Yaru/stereo/dialog-information.oga",
    )
    for path in sound_paths:
        if os.path.isfile(path) and shutil.which("paplay"):
            subprocess.run(["paplay", path], check=False, capture_output=True, timeout=30)
            return
    if shutil.which("canberra-gtk-play"):
        subprocess.run(
            ["canberra-gtk-play", "-f", "-i", "complete"],
            check=False,
            capture_output=True,
            timeout=10,
        )
        return
    print("\a", end="", flush=True)


# ================================================================
# Config
# ================================================================
def _resolve_vars(obj: Any, root: Dict) -> Any:
    """Recursively resolve ${section.key} references in config values."""
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        def _replace(match):
            keys = match.group(1).split(".")
            val = root
            for k in keys:
                val = val[k]
            return str(val)
        # Iterate until no more references (handles chained refs)
        for _ in range(5):
            new = pattern.sub(_replace, obj)
            if new == obj:
                break
            obj = new
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_vars(v, root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_vars(v, root) for v in obj]
    return obj


def load_config(path: str = "./config.yaml") -> Dict:
    """Load YAML config and resolve all ${var} interpolations."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_vars(raw, raw)


# ================================================================
# Date Utilities
# ================================================================
_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"]

def parse_date(date_str: Any) -> Optional[datetime]:
    """Parse a date string trying multiple common formats.

    Returns None on failure instead of raising an exception.
    """
    if isinstance(date_str, datetime):
        return date_str
    if date_str is None or pd.isna(date_str):
        return None
    s = str(date_str).split(" ")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def normalize_date(date_val: Any) -> Optional[str]:
    """Convert any date-like value to 'YYYY-MM-DD' string."""
    dt = parse_date(date_val)
    return dt.strftime("%Y-%m-%d") if dt else None


# ================================================================
# Data I/O
# ================================================================
def load_ticker_csv(directory: str, ticker: str, suffix: str = "_news.csv") -> pd.DataFrame:
    """Load a per-ticker CSV file from a directory."""
    path = os.path.join(directory, f"{ticker}{suffix}")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def list_tickers(directory: str, pattern: str = "*.csv") -> List[str]:
    """Extract unique ticker names from filenames in a directory."""
    files = glob.glob(os.path.join(directory, pattern))
    tickers = []
    for f in files:
        name = os.path.basename(f).split("_")[0].split(".")[0]
        if name and name not in tickers:
            tickers.append(name)
    return sorted(tickers)


def load_ticker_parquet(directory: str, ticker: str, suffix: str = ".parquet") -> pd.DataFrame:
    """Load a per-ticker Parquet file from a directory."""
    path = os.path.join(directory, f"{ticker}{suffix}")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


# ================================================================
# Text Deduplication
# ================================================================
def fuzzy_dedup(texts: List[str], threshold: float = 0.85) -> List[int]:
    """Return indices of unique texts after fuzzy deduplication.

    Uses SequenceMatcher ratio for pairwise similarity.
    Keeps the first occurrence when duplicates are found.
    """
    from difflib import SequenceMatcher

    keep = []
    for i, text_i in enumerate(texts):
        is_dup = False
        for j in keep:
            ratio = SequenceMatcher(None, text_i, texts[j]).ratio()
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return keep


# ================================================================
# DTW Ground Truth Loading
# ================================================================
def load_dtw_ground_truth(
    dtw_dir: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """Load DTW similarity CSV files and build a ground-truth dictionary.

    Returns:
        dict mapping (query_ticker, query_date) -> [(match_ticker, match_start_date), ...]
    """
    dtw_dict: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    for csv_file in glob.glob(os.path.join(dtw_dir, "*_sim*.csv")):
        try:
            df = pd.read_csv(csv_file)
            df["_q_dt"] = df["Query_Date"].apply(parse_date)
            df = df.dropna(subset=["_q_dt"])

            if start_date:
                df = df[df["_q_dt"] >= start_date]
            if end_date:
                df = df[df["_q_dt"] <= end_date]

            for _, row in df.iterrows():
                q_date_str = row["_q_dt"].strftime("%Y-%m-%d")
                key = (row["Query_Ticker"], q_date_str)

                m_dt = parse_date(row["Match_Start_Date"])
                m_date_str = m_dt.strftime("%Y-%m-%d") if m_dt else str(row["Match_Start_Date"]).split(" ")[0]

                dtw_dict.setdefault(key, []).append((row["Match_Ticker"], m_date_str))
        except Exception:
            continue

    return dtw_dict


# ================================================================
# Embedding Corpus Building
# ================================================================
def build_corpus_matrix(
    emb_dir: str,
) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[str, str, str]]]]:
    """Load parquet embeddings into a single numpy matrix.

    Returns:
        (embeddings_array, metadata_list) where metadata is [(ticker, start, end), ...]
        Returns (None, None) if no data found.
    """
    embeddings_list = []
    metadata = []

    for pq_file in glob.glob(os.path.join(emb_dir, "*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            if "embedding" not in df.columns or df.empty:
                continue

            for _, row in df.iterrows():
                start_dt = parse_date(row.get("Window_Start"))
                end_dt = parse_date(row.get("Window_End"))
                if not start_dt or not end_dt:
                    continue

                metadata.append((
                    row["ticker"],
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                ))
                embeddings_list.append(np.array(row["embedding"], dtype=np.float32))
        except Exception:
            continue

    if not embeddings_list:
        return None, None

    return np.stack(embeddings_list), metadata


# ================================================================
# Market Context Builder (Early Fusion)
# ================================================================
def load_price_cache(price_dir: str) -> Dict[str, pd.DataFrame]:
    """Pre-load all ticker price CSVs into a dict for fast repeated lookups.

    Returns:
        {ticker: DataFrame with columns [Date (datetime), Close (float), Volume (float)]}
        sorted by Date ascending.
    """
    cache: Dict[str, pd.DataFrame] = {}
    for fpath in glob.glob(os.path.join(price_dir, "*_daily.csv")):
        ticker = os.path.basename(fpath).replace("_daily.csv", "")
        try:
            df = pd.read_csv(fpath)
            date_col = next(
                (c for c in ["Date", "date", "timestamp"] if c in df.columns), None
            )
            if date_col is None:
                continue
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            else:
                df["Volume"] = np.nan
            df = df.sort_values("Date").reset_index(drop=True)
            cache[ticker] = df[["Date", "Close", "Volume"]]
        except Exception:
            continue
    return cache


def _pct_change(closes: np.ndarray) -> Optional[float]:
    """Return percentage change from first to last value, or None."""
    if len(closes) < 2 or closes[0] == 0 or not np.all(np.isfinite(closes)):
        return None
    return float((closes[-1] / closes[0] - 1) * 100)


def build_market_context(
    ticker: str,
    anchor_date,
    price_cache: Dict[str, pd.DataFrame],
    index_ticker: str = "QQQ",
    lookback_short: int = 5,
    lookback_long: int = 30,
) -> str:
    """Build a market-environment context string for LLM prompt injection.

    Uses only data on or before ``anchor_date`` (no lookahead bias).

    Args:
        ticker: Individual stock ticker (e.g. "AAPL").
        anchor_date: Window end date (str or datetime). Only data <= this date is used.
        price_cache: Pre-loaded price DataFrames from ``load_price_cache``.
        index_ticker: Market index ETF ticker (default "QQQ").
        lookback_short: Short-term lookback in trading days (default 5).
        lookback_long: Long-term lookback in trading days (default 30).

    Returns:
        Formatted context string (empty string if all data is missing).
    """
    anchor_dt = parse_date(anchor_date)
    if anchor_dt is None:
        return ""
    # Ensure anchor_dt is tz-naive for comparison
    anchor_dt = pd.Timestamp(anchor_dt).tz_localize(None)

    lines: List[str] = []

    def _ticker_stats(t: str) -> Optional[Dict[str, Any]]:
        df = price_cache.get(t)
        if df is None or df.empty:
            return None
        RetModel = df[df["Date"] <= anchor_dt]
        if len(RetModel) < lookback_short + 1:
            return None
        closes = RetModel["Close"].to_numpy()
        volumes = RetModel["Volume"].to_numpy()
        stats: Dict[str, Any] = {"ticker": t}

        # Short-term return
        short_closes = closes[-(lookback_short + 1):]
        stats["short_pct"] = _pct_change(short_closes)

        # Long-term return
        if len(closes) >= lookback_long + 1:
            long_closes = closes[-(lookback_long + 1):]
            stats["long_pct"] = _pct_change(long_closes)
        else:
            stats["long_pct"] = None

        # Volume deviation: short-term avg vs long-term avg
        if len(volumes) >= lookback_long and np.any(np.isfinite(volumes[-lookback_long:])):
            vol_short = np.nanmean(volumes[-lookback_short:])
            vol_long = np.nanmean(volumes[-lookback_long:])
            if vol_long > 0 and np.isfinite(vol_short):
                stats["vol_dev_pct"] = float((vol_short / vol_long - 1) * 100)
                stats["vol_short"] = float(vol_short)
                stats["vol_long"] = float(vol_long)
            else:
                stats["vol_dev_pct"] = None
        else:
            stats["vol_dev_pct"] = None

        return stats

    idx_stats = _ticker_stats(index_ticker)
    stk_stats = _ticker_stats(ticker)

    if idx_stats is None and stk_stats is None:
        return ""

    lines.append("[Market Context]")

    # Index line
    if idx_stats is not None:
        parts = []
        if idx_stats["short_pct"] is not None:
            parts.append(f"past {lookback_short} trading days: {idx_stats['short_pct']:+.1f}%")
        if idx_stats["long_pct"] is not None:
            parts.append(f"past {lookback_long} trading days: {idx_stats['long_pct']:+.1f}%")
        if parts:
            lines.append(f"- {index_ticker} (Nasdaq-100 ETF): {' | '.join(parts)}")

    # Stock line
    if stk_stats is not None:
        parts = []
        short_str = ""
        if stk_stats["short_pct"] is not None:
            short_str = f"past {lookback_short} trading days: {stk_stats['short_pct']:+.1f}%"
            # Relative outperformance vs index
            if idx_stats is not None and idx_stats["short_pct"] is not None:
                excess = stk_stats["short_pct"] - idx_stats["short_pct"]
                short_str += f" (vs {index_ticker}: {excess:+.1f}%)"
            parts.append(short_str)
        if stk_stats["long_pct"] is not None:
            long_str = f"past {lookback_long} trading days: {stk_stats['long_pct']:+.1f}%"
            if idx_stats is not None and idx_stats["long_pct"] is not None:
                excess = stk_stats["long_pct"] - idx_stats["long_pct"]
                long_str += f" (vs {index_ticker}: {excess:+.1f}%)"
            parts.append(long_str)
        if parts:
            lines.append(f"- {ticker}: {' | '.join(parts)}")

        # Volume
        if stk_stats.get("vol_dev_pct") is not None:
            vol_s = stk_stats["vol_short"] / 1e6
            vol_l = stk_stats["vol_long"] / 1e6
            lines.append(
                f"- {ticker} {lookback_short}-day avg volume: {vol_s:.1f}M "
                f"(vs {lookback_long}-day avg: {vol_l:.1f}M, {stk_stats['vol_dev_pct']:+.1f}%)"
            )

    return "\n".join(lines)