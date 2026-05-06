"""
Step 04 — Generate Multi-Branch Training Dataset
=================================================
Creates query-positive pairs based on DTW matches.
- AGG/SUM: Generates JSONL with text pairs.
- POOL: Generates Parquet with embedding pairs.
"""

import os
import json
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

from utility import get_logger, normalize_date, ensure_dir

logger = get_logger("step04")


def _run_text_pairs(conf: Dict):
    """(AGG/SUM) Build contrastive text pairs for MNRL.

    Each JSONL record contains:
        {"query": str, "pos": [str]}
    """
    data = conf["data"]
    branch = conf["experiment"]["NewsAgg"]
    
    dtw_dir = data["dtw_results_dir"]
    news_dir = data["news_5d_summarized_dir"]
    train_path = data["train_jsonl"]
    val_path = data["val_jsonl"]
    test_path = data["test_jsonl"]

    val_start = "2025-01-29"
    test_start = "2025-04-10"

    ensure_dir(os.path.dirname(train_path))
    sim_files = sorted([f for f in os.listdir(dtw_dir) if f.endswith("_similarity.csv")])

    n_train, n_val, n_test = 0, 0, 0
    with (
        open(train_path, "w", encoding="utf-8") as f_train,
        open(val_path, "w", encoding="utf-8") as f_val,
        open(test_path, "w", encoding="utf-8") as f_test,
    ):
        for sim_file in tqdm(sim_files, desc=f"Pairs ({branch})"):
            ticker = sim_file.replace("_similarity.csv", "")
            
            # Load news map
            news_csv = os.path.join(news_dir, f"{ticker}_5d_summaries.csv")
            if not os.path.exists(news_csv): continue
            ndf = pd.read_csv(news_csv)
            q_map = dict(zip(ndf["Window_End"].apply(normalize_date), ndf["LLM_5D_Trend_Summary"]))
            m_map = dict(zip(ndf["Window_End"].apply(normalize_date), ndf["LLM_5D_Trend_Summary"]))

            df = pd.read_csv(os.path.join(dtw_dir, sim_file))
            for _, row in df.iterrows():
                q_date, m_ticker, m_date = normalize_date(row["Query_Date"]), row["Match_Ticker"], normalize_date(row["Match_Start_Date"])
                
                # Context for match
                m_news_csv = os.path.join(news_dir, f"{m_ticker}_5d_summaries.csv")
                if not os.path.exists(m_news_csv): continue
                mdf = pd.read_csv(m_news_csv)
                match_text_map = dict(zip(mdf["Window_End"].apply(normalize_date), mdf["LLM_5D_Trend_Summary"]))
                
                q_text = q_map.get(q_date)
                p_text = match_text_map.get(m_date)
                
                if not (q_text and p_text): continue

                rec = json.dumps({"query": q_text, "pos": [p_text]}, ensure_ascii=False) + "\n"
                
                if q_date < val_start:
                    f_train.write(rec)
                    n_train += 1
                elif q_date < test_start:
                    f_val.write(rec)
                    n_val += 1
                else:
                    f_test.write(rec)
                    n_test += 1

    logger.info(f"Text pairs generated: train={n_train}, val={n_val}, test={n_test}")


def _run_pool_pairs(conf: Dict):
    """(POOL) Build vector-to-vector training pairs."""
    data = conf["data"]
    dtw_dir = data["dtw_results_dir"]
    news_dir = data["news_5d_summarized_dir"]
    train_out = data["train_jsonl"].replace(".jsonl", ".parquet")
    val_out = data["val_jsonl"].replace(".jsonl", ".parquet")
    
    val_start = "2025-01-29"
    test_start = "2025-04-10"

    sim_files = sorted([f for f in os.listdir(dtw_dir) if f.endswith("_similarity.csv")])
    train_recs, val_recs = [], []

    for sim_file in tqdm(sim_files, desc="Pairs (POOL)"):
        ticker = sim_file.replace("_similarity.csv", "")
        news_pq = os.path.join(news_dir, f"{ticker}_5d_pool.parquet")
        if not os.path.exists(news_pq): continue
        ndf = pd.read_parquet(news_pq)
        
        # Determine column: 'embedding' (max/avg) or 'embedding_seq' (attention)
        emb_col = "embedding_seq" if "embedding_seq" in ndf.columns else "embedding"
        q_map = dict(zip(ndf["Window_End"].apply(normalize_date), ndf[emb_col]))
        
        df = pd.read_csv(os.path.join(dtw_dir, sim_file))
        for _, row in df.iterrows():
            q_date, m_ticker, m_date = normalize_date(row["Query_Date"]), row["Match_Ticker"], normalize_date(row["Match_Start_Date"])
            
            m_news_pq = os.path.join(news_dir, f"{m_ticker}_5d_pool.parquet")
            if not os.path.exists(m_news_pq): continue
            mdf = pd.read_parquet(m_news_pq)
            m_emb_col = "embedding_seq" if "embedding_seq" in mdf.columns else "embedding"
            m_map = dict(zip(mdf["Window_End"].apply(normalize_date), mdf[m_emb_col]))
            
            q_val = q_map.get(q_date)
            p_val = m_map.get(m_date)
            
            if q_val is None or p_val is None: continue
            
            q_key = "query_seq" if "seq" in emb_col else "query_vec"
            p_key = "pos_seq" if "seq" in m_emb_col else "pos_vec"
            
            rec = {q_key: q_val, p_key: p_val, "ticker": ticker, "q_date": q_date}
            if q_date < val_start: train_recs.append(rec)
            elif q_date < test_start: val_recs.append(rec)

    if train_recs: pd.DataFrame(train_recs).to_parquet(train_out)
    if val_recs: pd.DataFrame(val_recs).to_parquet(val_out)
    logger.info(f"POOL datasets saved: {train_out}, {val_out}")


def run(conf: Dict, RetModel: Optional[str] = None):
    branch = conf["experiment"]["NewsAgg"]
    if branch in ["agg", "sum"]:
        _run_text_pairs(conf)
    else:
        _run_pool_pairs(conf)
