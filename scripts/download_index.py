#!/usr/bin/env python3
"""
scripts/download_index.py — Download Nasdaq-100 Index ETF (QQQ) Daily Data
===========================================================================
Downloads QQQ daily OHLCV data from yfinance, matching the date range and
column format of existing individual stock files in the prices directory.

Usage:
    cd /mnt/raid1/ken/Capstone_data/Capstone_Context
    python scripts/download_index.py

Output:
    Datasets/nasdaq100_prices_5yrs_yfinance/QQQ_daily.csv
"""

import os
import sys
import argparse

import yfinance as yf
import pandas as pd

# Add project root to path for utility imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Download QQQ daily data")
    parser.add_argument(
        "--ticker", type=str, default="QQQ",
        help="Index ETF ticker to download (default: QQQ)"
    )
    parser.add_argument(
        "--start", type=str, default="2021-01-01",
        help="Start date (default: 2021-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2025-12-31",
        help="End date (default: 2025-12-31)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Datasets", "nasdaq100_prices_5yrs_yfinance"
        ),
        help="Output directory for the CSV"
    )
    args = parser.parse_args()

    out_path = os.path.join(args.output_dir, f"{args.ticker}_daily.csv")

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        print(f"[INFO] {out_path} already exists ({len(existing)} rows). Skipping.")
        print(f"       Delete the file to force re-download.")
        return

    print(f"[INFO] Downloading {args.ticker} from {args.start} to {args.end} ...")
    df = yf.download(args.ticker, start=args.start, end=args.end, progress=True)

    if df.empty:
        print(f"[ERROR] No data returned for {args.ticker}. Check ticker / date range.")
        sys.exit(1)

    # Match the column format of existing individual stock CSVs:
    # Date,Open,High,Low,Close,Volume
    # Handle both single-level and multi-level column index from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in keep_cols:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found in downloaded data. Available: {list(df.columns)}")

    df = df[[c for c in keep_cols if c in df.columns]]
    df.index.name = "Date"
    df.to_csv(out_path)

    print(f"[OK] Saved {len(df)} rows to {out_path}")
    print(f"     Date range: {df.index.min()} — {df.index.max()}")


if __name__ == "__main__":
    main()
