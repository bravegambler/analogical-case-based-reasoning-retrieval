"""
scripts/data_download.py — Nasdaq-100 Data Download & Anomaly Detection
========================================================================
Combines three original notebooks into one runnable script:
  1. prices   — Download 5-year OHLCV data via Yahoo Finance
  2. anomalies — Detect price anomalies (Z-score > 2) per stock
  3. news     — Download news articles via Polygon.io API

Usage
-----
# Run all three steps in sequence:
    python scripts/data_download.py --step all --output_dir /path/to/Datasets

# Run individual steps:
    python scripts/data_download.py --step prices   --output_dir /path/to/Datasets
    python scripts/data_download.py --step anomalies --output_dir /path/to/Datasets
    python scripts/data_download.py --step news     --output_dir /path/to/Datasets \
        --polygon_api_key YOUR_KEY

# API key can also be set via environment variable (recommended):
    export POLYGON_API_KEY=your_key_here
    python scripts/data_download.py --step news --output_dir /path/to/Datasets

Output directories (created automatically under --output_dir):
    nasdaq100_prices_5yrs_yfinance/     <- OHLCV CSVs, one file per ticker
    nasdaq100_anomalies_per_stock/      <- Anomaly CSVs, one file per ticker
    nasdaq100_news_full/                <- News CSVs, one file per ticker
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

# Load .env file if present (pip install python-dotenv, or just export the var manually)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Ticker list
# ─────────────────────────────────────────────────────────────────────────────

NASDAQ_100_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI",  "ADP",  "ADSK", "AEP",  "ALNY", "AMAT", "AMD",
    "AMGN", "AMZN", "APP",  "ARM",  "ASML", "AVGO", "AXON", "AZN",  "BKR",  "BKNG",
    "CCEP", "CDNS", "CEG",  "CHTR", "CMCSA","COST", "CPRT", "CRWD", "CSCO", "CSGP",
    "CSX",  "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA",   "EXC",  "FANG",
    "FAST", "FER",  "FTNT", "GEHC", "GILD", "GOOG", "GOOGL","HON",  "IDXX", "INSM",
    "INTC", "INTU", "ISRG", "KDP",  "KHC",  "KLAC", "LIN",  "LRCX", "MAR",  "MCHP",
    "MDLZ", "MELI", "META", "MNST", "MPWR", "MSFT", "MSTR", "MU",   "NFLX", "NKE",
    "NXPI", "NVDA", "ODFL", "ORLY", "PANW", "PAYX", "PCAR", "PEP",  "PLTR", "PDD",
    "PYPL", "QCOM", "REGN", "ROP",  "ROST", "SBUX", "SIRI", "SNPS", "STX",  "TMUS",
    "TSLA", "TTWO", "TXN",  "VRTX", "VRSK", "WBD",  "WDC",  "WDAY", "XEL",  "ZS",
    "TRI",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: OHLCV download (Yahoo Finance)
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(output_dir: str, start_date: str, end_date: str,
                    tickers: list = None) -> None:
    """Download 5-year daily OHLCV data for all Nasdaq-100 tickers via yfinance."""
    save_dir = os.path.join(output_dir, "nasdaq100_prices_5yrs_yfinance")
    os.makedirs(save_dir, exist_ok=True)

    tickers = tickers or NASDAQ_100_TICKERS
    print(f"\n[prices] Downloading {len(tickers)} tickers → {save_dir}")
    print(f"[prices] Date range: {start_date} to {end_date}\n")

    success, skipped, failed = 0, 0, []

    for i, ticker in enumerate(tickers, 1):
        fpath = os.path.join(save_dir, f"{ticker}_daily.csv")

        if os.path.exists(fpath):
            print(f"  [{i}/{len(tickers)}] {ticker:<6}  already exists, skipping.")
            skipped += 1
            continue

        print(f"  [{i}/{len(tickers)}] {ticker:<6}  downloading...", end="", flush=True)
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=False)
            if df.empty:
                with open(fpath, "w") as f:
                    f.write("No Data")
                print(" → no data")
                failed.append(ticker)
                continue

            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df.to_csv(fpath, index=False)
            print(f" → {len(df)} rows saved")
            success += 1
        except Exception as exc:
            print(f" → ERROR: {exc}")
            failed.append(ticker)

        time.sleep(0.5)

    print(f"\n[prices] Done. success={success}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print(f"[prices] Failed tickers: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Anomaly detection (rolling Z-score)
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(output_dir: str, z_threshold: float = 2.0,
                     tickers: list = None) -> None:
    """
    Compute rolling Z-scores on daily returns and flag anomalies (|Z| > threshold).
    Uses a 252-trading-day rolling window (min 63 days).
    """
    input_dir  = os.path.join(output_dir, "nasdaq100_prices_5yrs_yfinance")
    save_dir   = os.path.join(output_dir, "nasdaq100_anomalies_per_stock")
    os.makedirs(save_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if tickers:
        csv_files = [f for f in all_files
                     if f.replace("_daily.csv", "").replace(".csv", "") in tickers]
    else:
        csv_files = all_files
    if not csv_files:
        print(f"[anomalies] No CSV files found in {input_dir}. Run --step prices first.")
        sys.exit(1)

    print(f"\n[anomalies] Processing {len(csv_files)} files → {save_dir}")
    print(f"[anomalies] Z-score threshold: ±{z_threshold}\n")

    saved, empty = 0, 0

    for i, fname in enumerate(sorted(csv_files), 1):
        ticker = fname.replace("_daily.csv", "").replace(".csv", "")
        fpath  = os.path.join(input_dir, fname)

        try:
            df = pd.read_csv(fpath)
            if df.empty or len(df) < 2 or "Close" not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df["Return"] = df["Close"].pct_change()
            df = df.dropna(subset=["Return"])

            rolling = df["Return"].rolling(window=252, min_periods=63)
            df["Rolling_Mean"] = rolling.mean()
            df["Rolling_Std"]  = rolling.std()
            df = df.dropna(subset=["Rolling_Std"])
            df = df[df["Rolling_Std"] != 0]
            df["Z_Score"] = (df["Return"] - df["Rolling_Mean"]) / df["Rolling_Std"]

            anomalies = df[df["Z_Score"].abs() > z_threshold].copy()

            if anomalies.empty:
                print(f"  [{i}/{len(csv_files)}] {ticker:<6}  no anomalies found")
                empty += 1
                continue

            anomalies["Return"]  = anomalies["Return"].round(4)
            anomalies["Z_Score"] = anomalies["Z_Score"].round(2)
            out_path = os.path.join(save_dir, f"{ticker}_anomalies.csv")
            anomalies.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  [{i}/{len(csv_files)}] {ticker:<6}  {len(anomalies)} anomalies saved")
            saved += 1

        except Exception as exc:
            print(f"  [{i}/{len(csv_files)}] {ticker:<6}  ERROR: {exc}")

    print(f"\n[anomalies] Done. saved={saved}  no_anomalies={empty}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: News download (Polygon.io)
# ─────────────────────────────────────────────────────────────────────────────

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
NEWS_COLUMNS     = ["published_utc", "title", "article_url", "description", "keywords", "author"]


def _fetch_news_for_ticker(ticker: str, api_key: str,
                           start_date: str, end_date: str,
                           page_sleep: float = 15.0) -> pd.DataFrame:
    """Fetch all news pages for a single ticker from Polygon.io."""
    all_results = []
    params = {
        "ticker":              ticker,
        "published_utc.gte":  start_date,
        "published_utc.lte":  end_date,
        "limit":               1000,
        "order":               "desc",
        "sort":                "published_utc",
        "apiKey":              api_key,
    }
    current_url = POLYGON_NEWS_URL
    page = 0

    while current_url:
        try:
            if current_url == POLYGON_NEWS_URL:
                resp = requests.get(current_url, params=params, timeout=30)
            else:
                sep = "&" if "?" in current_url else "?"
                url = current_url if "apiKey=" in current_url \
                      else f"{current_url}{sep}apiKey={api_key}"
                resp = requests.get(url, timeout=30)

            if resp.status_code == 429:
                print(f"\n    [rate limit] cooling down 60s (page {page})...", end="")
                time.sleep(60)
                continue

            if resp.status_code != 200:
                print(f"\n    [HTTP {resp.status_code}] {resp.text[:120]}")
                break

            data    = resp.json()
            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            page += 1
            print(f"\r    page {page}, total {len(all_results)} articles...", end="", flush=True)

            current_url = data.get("next_url")
            if current_url:
                time.sleep(page_sleep)

        except Exception as exc:
            print(f"\n    [exception] {exc}")
            break

    return pd.DataFrame(all_results)


def download_news(output_dir: str, api_key: str,
                  start_date: str, end_date: str,
                  page_sleep: float = 15.0,
                  tickers: list = None) -> None:
    """Download Polygon.io news for all Nasdaq-100 tickers."""
    save_dir = os.path.join(output_dir, "nasdaq100_news_full")
    os.makedirs(save_dir, exist_ok=True)

    tickers = tickers or NASDAQ_100_TICKERS
    print(f"\n[news] Downloading news for {len(tickers)} tickers → {save_dir}")
    print(f"[news] Date range: {start_date} to {end_date}")
    print(f"[news] Rate-limit sleep: {page_sleep}s per page\n")

    success, skipped = 0, 0

    for i, ticker in enumerate(tickers, 1):
        fpath = os.path.join(save_dir, f"{ticker}_news.csv")

        if os.path.exists(fpath):
            print(f"  [{i}/{len(tickers)}] {ticker:<6}  already exists, skipping.")
            skipped += 1
            continue

        print(f"  [{i}/{len(tickers)}] {ticker:<6}  fetching...", end="", flush=True)
        df = _fetch_news_for_ticker(ticker, api_key, start_date, end_date, page_sleep)

        if df.empty:
            with open(fpath, "w") as f:
                f.write("No Data")
            print(f"\n    → no articles found")
        else:
            cols = [c for c in NEWS_COLUMNS if c in df.columns]
            df[cols].to_csv(fpath, index=False, encoding="utf-8-sig")
            print(f"\n    → {len(df)} articles saved")
            success += 1

        time.sleep(page_sleep)

    print(f"\n[news] Done. success={success}  skipped={skipped}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    _5y_ago = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    _today  = datetime.now().strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Nasdaq-100 data download pipeline (prices + anomalies + news)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", required=True,
        choices=["prices", "anomalies", "news", "all"],
        help="Which step(s) to run.",
    )
    parser.add_argument(
        "--output_dir", default="./Datasets",
        help="Root output directory. Sub-folders are created automatically. "
             "(default: ./Datasets)",
    )
    parser.add_argument(
        "--start_date", default=_5y_ago,
        help=f"Download start date YYYY-MM-DD (default: 5 years ago = {_5y_ago})",
    )
    parser.add_argument(
        "--end_date", default=_today,
        help=f"Download end date YYYY-MM-DD (default: today = {_today})",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None, metavar="TICKER",
        help="Limit download to specific tickers, e.g. --tickers AAPL MSFT. "
             "Default: all 101 Nasdaq-100 tickers.",
    )
    parser.add_argument(
        "--polygon_api_key", default=None,
        help="Polygon.io API key (required for --step news/all). "
             "Can also be set via POLYGON_API_KEY env var.",
    )
    parser.add_argument(
        "--z_threshold", type=float, default=2.0,
        help="Z-score threshold for anomaly detection (default: 2.0)",
    )
    parser.add_argument(
        "--page_sleep", type=float, default=15.0,
        help="Seconds to wait between Polygon.io pages (default: 15.0 for free tier)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_prices    = args.step in ("prices",    "all")
    run_anomalies = args.step in ("anomalies", "all")
    run_news      = args.step in ("news",      "all")

    if run_prices:
        download_prices(args.output_dir, args.start_date, args.end_date,
                        tickers=args.tickers)

    if run_anomalies:
        detect_anomalies(args.output_dir, z_threshold=args.z_threshold,
                         tickers=args.tickers)

    if run_news:
        api_key = args.polygon_api_key or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            print(
                "[news] ERROR: Polygon.io API key is required.\n"
                "  Pass --polygon_api_key YOUR_KEY  or  export POLYGON_API_KEY=YOUR_KEY"
            )
            sys.exit(1)
        download_news(
            args.output_dir, api_key,
            args.start_date, args.end_date,
            page_sleep=args.page_sleep,
            tickers=args.tickers,
        )

    print("\nAll requested steps completed.")


if __name__ == "__main__":
    main()
