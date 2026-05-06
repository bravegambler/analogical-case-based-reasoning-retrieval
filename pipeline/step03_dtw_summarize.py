"""
Step 03 — Multi-Branch Context Generation (AGG / SUM / POOL)
=============================================================
Implementation of three distinct experimental branches:
- AGG : Raw text aggregation.
- SUM : LLM summarization.
- POOL: Vector pooling.
"""

import os
import glob
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm # for progress bar


from utility import get_logger, ensure_dir, parse_date, fuzzy_dedup, load_ticker_csv, load_price_cache, build_market_context

logger = get_logger("step03")


# ================================================================
# Shared Logic: DTW Similarity
# ================================================================
def _compute_dtw(conf: Dict):
    """(Shared) Compute DTW distance for anomaly events."""
    from dtaidistance import dtw as dtw_distance

    data = conf["data"]
    dtw_conf = conf["dtw"]
    top_k = dtw_conf["top_k"]

    anomaly_dir = data["anomalies_per_stock_dir"]
    ts_dir = data["post_anomaly_ts_dir"]
    output_dir = ensure_dir(data["dtw_results_dir"])

    anomaly_files = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(".csv")])
    logger.info(f"Computing DTW Top-{top_k} for {len(anomaly_files)} stocks")

    candidates = []
    for af in tqdm(anomaly_files, desc="Loading candidates"):
        ticker = af.replace("_anomalies.csv", "")
        ts_path = os.path.join(ts_dir, f"{ticker}_post_moves.csv")
        if not os.path.exists(ts_path): continue
        try:
            df = pd.read_csv(ts_path)
            for anom_date, group in df.groupby("anomaly_date"):
                prices = group["Close"].values
                if len(prices) >= 5:
                    # Z-normalize matching the previous experimental logic
                    norm = (prices / prices[0] - 1) * 100
                    candidates.append((ticker, anom_date, norm.astype(np.double)))
        except Exception: continue

    for af in tqdm(anomaly_files, desc="DTW matching"):
        ticker = af.replace("_anomalies.csv", "")
        out_path = os.path.join(output_dir, f"{ticker}_similarity.csv")
        if os.path.exists(out_path): continue

        anomaly_path = os.path.join(anomaly_dir, af)
        try: df_a = pd.read_csv(anomaly_path)
        except Exception: continue

        results = []
        for _, row in df_a.iterrows():
            q_date = str(row.get("Date", ""))
            q_seq = next((c[2] for c in candidates if c[0] == ticker and c[1] == q_date), None)
            if q_seq is None: continue

            distances = []
            for c_ticker, c_date, c_seq in candidates:
                # Chronological Integrity: Only match past events
                if c_date >= q_date: continue 
                
                try:
                    # dtaidistance returns float directly
                    d = dtw_distance.distance(q_seq, c_seq)
                    distances.append((c_ticker, c_date, d))
                except Exception: continue

            distances.sort(key=lambda x: x[2])
            for rank, (m_ticker, m_date, dist) in enumerate(distances[:top_k], 1):
                results.append({
                    "Query_Ticker": ticker, "Query_Date": q_date, "Match_Rank": rank,
                    "Match_Ticker": m_ticker, "Match_Start_Date": m_date, "DTW_Distance": round(dist, 4)
                })
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)


# ================================================================
# Branch: AGG (Raw Aggregation)
# ================================================================
def _run_agg_branch(conf: Dict):
    """Experiment 1: Raw text daily aggregation + 5D windowing."""
    data = conf["data"]
    pp_conf = conf["preprocess"]
    news_dir = data["news_full_dir"]
    daily_out = ensure_dir(data["news_summarized_dir"])
    window_out = ensure_dir(data["news_5d_summarized_dir"])

    tickers = [f.replace("_news.csv", "") for f in os.listdir(news_dir) if f.endswith("_news.csv")]
    logger.info(f"AGG Branch: Processing {len(tickers)} tickers")

    for ticker in tqdm(tickers, desc="AGG Daily"):
        daily_path = os.path.join(daily_out, f"{ticker}_daily_summaries.csv")
        if not os.path.exists(daily_path):
            df = load_ticker_csv(news_dir, ticker)
            if df.empty: continue
            
            date_col = next((c for c in ["published_utc", "timestamp", "date"] if c in df.columns), None)
            if not date_col: continue
            df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            
            daily_rows = []
            for date, group in df.groupby("Date"):
                titles = group["title"].fillna("").astype(str).tolist()
                keep = fuzzy_dedup(titles, threshold=pp_conf["similarity_threshold"])
                agg_text = " | ".join([titles[i] for i in keep])
                daily_rows.append({"ticker": ticker, "Date": date, "LLM_Daily_Summary": agg_text})
            
            if daily_rows: pd.DataFrame(daily_rows).to_csv(daily_path, index=False)

    for ticker in tqdm(tickers, desc="AGG 5D Window"):
        _build_5d_window_text(ticker, daily_out, data["prices_dir"], window_out)

def _build_5d_window_text(ticker, daily_dir, price_dir, out_dir):
    out_path = os.path.join(out_dir, f"{ticker}_5d_summaries.csv")
    if os.path.exists(out_path): return

    daily_csv = os.path.join(daily_dir, f"{ticker}_daily_summaries.csv")
    price_csv = os.path.join(price_dir, f"{ticker}_daily.csv")
    if not (os.path.exists(daily_csv) and os.path.exists(price_csv)): return

    news_df = pd.read_csv(daily_csv)
    price_df = pd.read_csv(price_csv)
    p_date_col = next((c for c in ["Date", "date", "timestamp"] if c in price_df.columns), None)
    if not p_date_col:
        logger.warning(f"Skipping {ticker}: No Date column found in {price_csv}")
        return

    news_df["Date"] = pd.to_datetime(news_df["Date"])
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    tds = sorted(price_df["Date"].unique())
    
    td_df = pd.DataFrame({"Trading_Date": tds})
    merged = pd.merge_asof(news_df.sort_values("Date"), td_df, left_on="Date", right_on="Trading_Date", direction="forward")

    results = []
    for i in range(4, len(tds)):
        w_tds = tds[i-4 : i+1]
        w_news = merged[merged["Trading_Date"].isin(w_tds)]
        if w_news.empty: continue
        
        concat_text = " || ".join(w_news["LLM_Daily_Summary"].astype(str))
        results.append({
            "ticker": ticker, "Window_Start": tds[i-4].strftime("%Y-%m-%d"),
            "Window_End": tds[i].strftime("%Y-%m-%d"), "LLM_5D_Trend_Summary": concat_text
        })
    if results: 
        pd.DataFrame(results).to_csv(out_path, index=False)


# ================================================================
# Branch: SUM (LLM Summarization)
# ================================================================
def _run_sum_branch(conf: Dict):
    """Experiment 2: vLLM Daily Summary -> vLLM 5D Trend Synthesis.

    Early Fusion: injects market context (QQQ index + stock price/volume
    trends) into the LLM prompt so the generated summaries internalise
    market-environment information alongside news semantics.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = conf["data"]
    s_conf = conf["summarize"]
    
    d_conf = conf.get("device", {})
    llm = LLM(
        model=s_conf["model_name"], 
        trust_remote_code=True, 
        gpu_memory_utilization=d_conf.get("vllm_gpu_utilization", 0.9), 
        tensor_parallel_size=d_conf.get("vllm_tensor_parallel", 1),
        dtype="bfloat16",
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(s_conf["model_name"], trust_remote_code=True)
    sampling_params = SamplingParams(temperature=s_conf["temperature"], max_tokens=s_conf["max_tokens"])

    # Pre-load all price data once for market context injection
    price_dir = data["prices_dir"]
    logger.info(f"Loading price cache from {price_dir} for market context injection...")
    price_cache = load_price_cache(price_dir)
    logger.info(f"Price cache loaded: {len(price_cache)} tickers (QQQ present: {'QQQ' in price_cache})")

    # 1. Daily Summary (with market context)
    _do_llm_daily(data, llm, tokenizer, sampling_params, price_cache)
    # 2. 5D Trend (with market context)
    _do_llm_5d_trend(data, llm, tokenizer, sampling_params, price_cache)

def _do_llm_daily(data, llm, tokenizer, sampling_params, price_cache):
    input_dir = data["news_full_dir"]
    output_dir = ensure_dir(data["news_summarized_dir"])
    tickers = [f.replace("_news.csv", "") for f in os.listdir(input_dir) if f.endswith("_news.csv")]
    
    for ticker in tqdm(tickers, desc="SUM Daily LLM"):
        out_path = os.path.join(output_dir, f"{ticker}_daily_summaries.csv")
        if os.path.exists(out_path): continue
        
        df = load_ticker_csv(input_dir, ticker)
        if df.empty: continue
        date_col = next((c for c in ["published_utc", "timestamp", "date"] if c in df.columns), None)
        if not date_col: continue
        df["_day"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        
        prompts, metadata = [], []
        for d, group in df.groupby("_day"):
            text = " | ".join(group["title"].fillna("").astype(str).tolist()[:50]) # limit per day
            ctx = build_market_context(ticker, d, price_cache)
            p = (
                f"You are a senior equity analyst. Below is the market context and news for {ticker} on {d}.\n\n"
                f"{ctx}\n\n"
                f"[News Headlines]\n{text}\n\n"
                f"Task: Write a concise daily brief (2-3 sentences) that explains "
                f"how these news events interact with the current market environment. "
                f"Note whether {ticker} is moving with or against the broader Nasdaq-100, "
                f"and what drivers are at play."
            )
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
            metadata.append({"ticker": ticker, "Date": d})
        
        if not prompts: continue
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        recs = [{"ticker": m["ticker"], "Date": m["Date"], "LLM_Daily_Summary": o.outputs[0].text} for o, m in zip(outputs, metadata)]
        pd.DataFrame(recs).to_csv(out_path, index=False)

def _do_llm_5d_trend(data, llm, tokenizer, sampling_params, price_cache):
    daily_dir = data["news_summarized_dir"]
    price_dir = data["prices_dir"]
    output_dir = ensure_dir(data["news_5d_summarized_dir"])
    
    tickers = [f.replace("_daily_summaries.csv", "") for f in os.listdir(daily_dir) if f.endswith("_daily_summaries.csv")]
    for ticker in tqdm(tickers, desc="SUM 5D Trend LLM"):
        out_path = os.path.join(output_dir, f"{ticker}_5d_summaries.csv")
        if os.path.exists(out_path): continue
        
        price_csv = os.path.join(price_dir, f"{ticker}_daily.csv")
        if not os.path.exists(price_csv): continue
        
        news_df = pd.read_csv(os.path.join(daily_dir, f"{ticker}_daily_summaries.csv"))
        price_df = pd.read_csv(price_csv)
        news_df["Date"] = pd.to_datetime(news_df["Date"])
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        tds = sorted(price_df["Date"].unique())
        
        td_df = pd.DataFrame({"Trading_Date": tds})
        merged = pd.merge_asof(news_df.sort_values("Date"), td_df, left_on="Date", right_on="Trading_Date", direction="forward")
        
        prompts, metadata = [], []
        for i in range(4, len(tds)):
            w_tds = tds[i-4 : i+1]
            w_news = merged[merged["Trading_Date"].isin(w_tds)]
            if w_news.empty: continue
            
            compiled = " | ".join(w_news["LLM_Daily_Summary"].astype(str))
            ctx = build_market_context(ticker, tds[i], price_cache)
            p = (
                f"You are a senior equity analyst. Below is the market context and 5-day news summary for {ticker} "
                f"from {tds[i-4].strftime('%Y-%m-%d')} to {tds[i].strftime('%Y-%m-%d')}.\n\n"
                f"{ctx}\n\n"
                f"[5-Day News Summary]\n{compiled}\n\n"
                f"Task: Synthesize a comprehensive 5-day trend analysis (3-5 sentences). "
                f"Cover: (1) How did {ticker} perform relative to the Nasdaq-100? "
                f"(2) What news events drove the divergence or convergence? "
                f"(3) Is the current price action consistent with the news sentiment, or is there a disconnect?"
            )
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
            metadata.append({"ticker": ticker, "Start": tds[i-4], "End": tds[i]})
        
        if not prompts: continue
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        recs = [{"ticker": m["ticker"], "Window_Start": m["Start"], "Window_End": m["End"], "LLM_5D_Trend_Summary": o.outputs[0].text} for o, m in zip(outputs, metadata)]
        pd.DataFrame(recs).to_csv(out_path, index=False)


# ================================================================
# Branch: POOL (Vector Pooling)
# ================================================================
def _run_pool_branch(conf: Dict):
    """Experiment 3: Raw News Embedding -> Daily/5D Pooling."""
    import torch
    from sentence_transformers import SentenceTransformer
    
    data = conf["data"]
    p_conf = conf["pool"]
    method = p_conf["pooling_type"]
    
    news_dir = data["news_full_dir"]
    daily_out = ensure_dir(data["news_summarized_dir"])
    window_out = ensure_dir(data["news_5d_summarized_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        p_conf["embedding_model"],
        device=device,
        model_kwargs={"use_safetensors": True},
    )

    tickers = [f.replace("_news.csv", "") for f in os.listdir(news_dir) if f.endswith("_news.csv")]
    for ticker in tqdm(tickers, desc="POOL Embedding & Daily"):
        out_path = os.path.join(daily_out, f"{ticker}_daily_pool.parquet")
        if os.path.exists(out_path): continue
        
        df = load_ticker_csv(news_dir, ticker)
        if df.empty: continue
        
        text_col = next((c for c in ["title", "description"] if c in df.columns), df.columns[-1])
        texts = df[text_col].fillna("").astype(str).tolist()
        embs = model.encode(texts, batch_size=p_conf.get("batch_size", 64), convert_to_numpy=True)
        
        df["embedding"] = list(embs)
        df["Date"] = pd.to_datetime(df.get("published_utc", df.get("date", "2021-01-01"))).dt.strftime("%Y-%m-%d")
        
        daily_records = []
        for date, group in df.groupby("Date"):
            matrix = np.stack(group["embedding"].tolist())
            # We save the raw matrix (stack of article embs) to allow trainable pooling
            # But to save space/time, for MAX/AVG we can pre-pool. 
            # For ATTENTION, we MUST save the stack.
            if method in ["max", "avg"]:
                if method == "max": pooled = np.max(matrix, axis=0)
                else: pooled = np.mean(matrix, axis=0)
                daily_records.append({"ticker": ticker, "Date": date, "embedding": pooled.tolist()})
            else:
                # 'attention' or other trainable methods: save the list of vectors
                daily_records.append({"ticker": ticker, "Date": date, "embedding_seq": matrix.tolist()})
        
        if daily_records: pd.DataFrame(daily_records).to_parquet(out_path)

    for ticker in tqdm(tickers, desc="POOL 5D Window"):
        _build_5d_window_pool(ticker, daily_out, data["prices_dir"], window_out, method)

def _build_5d_window_pool(ticker, daily_dir, price_dir, out_dir, method):
    """POOL 5D window from DAILY-POOLED vectors.

    Source: {ticker}_daily_pool.parquet produced by `_run_pool_branch`, i.e. each row is
    a day whose vector is already an article-level pool (max/avg) or a list of per-article
    vectors (embedding_seq).
    Window: 5 consecutive trading days (tds[i-4 .. i]).
    Output: {ticker}_5d_pool.parquet
      - "embedding" path   : re-pool the 5 daily vectors by max/avg (pool twice overall)
      - "embedding_seq" path: concatenate daily per-article sequences (no re-pooling)

    NOTE: This is the article -> day -> 5D (vector-space) variant. A text-space variant
    (`_run_pool_text5d`) encodes AGG/SUM's LLM_5D_Trend_Summary directly into one vector
    per window and writes to a separate output directory.
    """
    out_path = os.path.join(out_dir, f"{ticker}_5d_pool.parquet")
    if os.path.exists(out_path): return

    daily_pq = os.path.join(daily_dir, f"{ticker}_daily_pool.parquet")
    price_csv = os.path.join(price_dir, f"{ticker}_daily.csv")
    if not (os.path.exists(daily_pq) and os.path.exists(price_csv)): return

    news_df = pd.read_parquet(daily_pq)
    price_df = pd.read_csv(price_csv)
    news_df["Date"] = pd.to_datetime(news_df["Date"])
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    tds = sorted(price_df["Date"].unique())
    
    td_df = pd.DataFrame({"Trading_Date": tds})
    merged = pd.merge_asof(news_df.sort_values("Date"), td_df, left_on="Date", right_on="Trading_Date", direction="forward")

    results = []
    emb_col = "embedding_seq" if "embedding_seq" in news_df.columns else "embedding"
    
    for i in range(4, len(tds)):
        w_tds = tds[i-4 : i+1]
        w_news = merged[merged["Trading_Date"].isin(w_tds)]
        if w_news.empty: continue
        
        # Flatten the list of vectors/matrices if needed
        # If max/avg: matrix is (news_count, dim)
        # If attention: matrix is (news_count, articles_count, dim) -> needs flattening for window pooling
        matrices = w_news[emb_col].tolist()
        
        if emb_col == "embedding":
            # [N, Dim]
            matrix = np.stack(matrices)
            if method == "max": win_emb = np.max(matrix, axis=0).tolist()
            else: win_emb = np.mean(matrix, axis=0).tolist()
            results.append({
                "ticker": ticker, "Window_Start": tds[i-4].strftime("%Y-%m-%d"),
                "Window_End": tds[i].strftime("%Y-%m-%d"), "embedding": win_emb
            })
        else:
            # emb_col == "embedding_seq"
            # Each record is a list of lists of floats.
            # We want to keep it as a sequence for the window-level trainer
            # Just concatenate all daily article lists into one long window-level sequence
            combined_seq = []
            for daily_seq in matrices:
                combined_seq.extend(daily_seq)
            
            results.append({
                "ticker": ticker, "Window_Start": tds[i-4].strftime("%Y-%m-%d"),
                "Window_End": tds[i].strftime("%Y-%m-%d"), "embedding_seq": combined_seq
            })
    
    if results: pd.DataFrame(results).to_parquet(out_path)


# ================================================================
# Branch: POOL (Text-based 5D variant)
# ================================================================
def _run_pool_text5d(conf: Dict):
    """POOL 5D window by encoding existing 5D TEXT (AGG or SUM) into one vector per window.

    Difference vs `_run_pool_branch` / `_build_5d_window_pool`:
      - Vector-space variant (existing): article -> day-pool -> 5D-pool (pool twice).
      - Text-space variant (this fn)   : pre-built 5D text -> single vector per window.

    Inputs  (read-only, reuses existing artifacts):
        Dataset root / nasdaq100_5d_summaries_{text_source}_{EmbModel} / {ticker}_5d_summaries.csv
        with columns [ticker, Window_Start, Window_End, LLM_5D_Trend_Summary].
    Output (new, separate from daily-pool output so both variants can coexist):
        Dataset root / nasdaq100_5d_pool_text_{text_source}_{EmbModel} / {ticker}_5d_pool.parquet
        with columns [ticker, Window_Start, Window_End, embedding].

    Usage:
        python main.py --step 03 --NewsAgg pool --RetModel text

    Requires that the chosen `text_source` branch (agg or sum) has already produced its
    5D text outputs (Step 03 on that branch).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    data = conf["data"]
    p_conf = conf["pool"]
    EmbModel = conf["experiment"]["EmbModel"]
    text_source = p_conf.get("text5d_source", "sum")
    if text_source not in ("agg", "sum"):
        raise ValueError(f"pool.text5d_source must be 'agg' or 'sum', got: {text_source!r}")

    dataset_dir = data["dataset_dir"]
    src_dir = os.path.join(dataset_dir, f"nasdaq100_5d_summaries_{text_source}_{EmbModel}")
    out_dir = ensure_dir(os.path.join(dataset_dir, f"nasdaq100_5d_pool_text_{text_source}_{EmbModel}"))

    if not os.path.isdir(src_dir):
        logger.error(
            f"5D text source not found: {src_dir}. "
            f"Run Step 03 on branch '{text_source}' first to produce *_5d_summaries.csv."
        )
        return

    csv_files = sorted(glob.glob(os.path.join(src_dir, "*_5d_summaries.csv")))
    if not csv_files:
        logger.warning(f"No *_5d_summaries.csv found in {src_dir}. Nothing to encode.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        p_conf["embedding_model"],
        device=device,
        model_kwargs={"use_safetensors": True},
    )
    batch_size = p_conf.get("batch_size", 64)

    logger.info(
        f"POOL (text5d): encoding {len(csv_files)} tickers from branch='{text_source}' "
        f"using embedder='{p_conf['embedding_model']}' on device='{device}'"
    )

    for fpath in tqdm(csv_files, desc=f"POOL text5d ({text_source})"):
        ticker = os.path.basename(fpath).replace("_5d_summaries.csv", "")
        out_path = os.path.join(out_dir, f"{ticker}_5d_pool.parquet")
        if os.path.exists(out_path):
            continue

        try:
            df = pd.read_csv(fpath)
            if df.empty or "LLM_5D_Trend_Summary" not in df.columns:
                continue
            if not {"Window_Start", "Window_End"}.issubset(df.columns):
                logger.warning(f"Skip {ticker}: missing Window_Start/Window_End in {fpath}")
                continue

            texts = df["LLM_5D_Trend_Summary"].fillna("").astype(str).tolist()
            embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

            out_df = pd.DataFrame({
                "ticker": ticker,
                "Window_Start": df["Window_Start"].astype(str),
                "Window_End": df["Window_End"].astype(str),
                "embedding": list(embs),
            })
            out_df.to_parquet(out_path)
        except Exception as e:
            logger.warning(f"Error encoding {ticker} ({fpath}): {e}")


def run(conf: Dict, RetModel: Optional[str] = None):
    branch = conf["experiment"]["NewsAgg"]
    if RetModel == "dtw":
        _compute_dtw(conf)
    elif branch == "agg":
        _run_agg_branch(conf)
    elif branch == "sum":
        _run_sum_branch(conf)
    elif branch == "pool":
        if RetModel in ("text", "text5d"):
            _run_pool_text5d(conf)
        else:
            _run_pool_branch(conf)
    else:
        raise ValueError(f"Unknown branch: {branch}")
