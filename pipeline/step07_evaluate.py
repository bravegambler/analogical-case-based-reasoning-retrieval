"""
Step 07 — Evaluation: Recall@10 & NDCG@10
=========================================
Performs benchmarks against DTW-based Ground Truth (GT).
Supports multiple retrieval methods and cross-method error analysis.

Sub-steps:
  bge        : Evaluate fine-tuned BGE-M3 (dense retrieval)
  bm25       : Evaluate classical BM25 (lexical retrieval)
  contriever : Evaluate Facebook Contriever (zero-shot baseline)
  cross      : Cross-method overlap and error analysis

Usage:
  python main.py --step 07 --RetModel bge
  python main.py --step 07 --RetModel cross
"""

import os
import glob
import re
import math
import json
import time
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utility import get_logger, parse_date, load_dtw_ground_truth
from models.Retriever import EmbeddingRetriever

logger = get_logger("step07")


# ================================================================
# BM25 Implementation (Lexical Baseline)
# ================================================================

class FastBM25:
    """Efficient BM25 implementation for lexical retrieval baseline."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(1, self.n)
        self.k1 = k1
        self.b = b
        
        self.inverted_index = {}
        for idx, doc in enumerate(corpus):
            for term, count in Counter(doc).items():
                self.inverted_index.setdefault(term, {})[idx] = count
                
        self.idf = {
            term: math.log(((self.n - len(docs) + 0.5) / (len(docs) + 0.5)) + 1.0)
            for term, docs in self.inverted_index.items()
        }
            
    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.n)
        for term in query:
            if term not in self.inverted_index:
                continue
            q_idf = self.idf[term]
            for doc_idx, freq in self.inverted_index[term].items():
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl))
                num = freq * (self.k1 + 1)
                scores[doc_idx] += q_idf * (num / denom)
        return scores


def tokenize(text: Any) -> List[str]:
    """Basic lexical tokenization."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())


# ================================================================
# Core Evaluation Loop
# ================================================================

def compute_metrics(
    retrieved: List[Tuple[str, str, float]], 
    gt_matches: List[Tuple[str, str]], 
    top_k: int = 10,
    proximity_days: int = 5
) -> Tuple[float, float, List[str]]:
    """Compute Recall and NDCG for a single query."""
    gt_list = gt_matches[:top_k]
    gt_parsed = [(t, parse_date(d)) for t, d in gt_list if parse_date(d)]
    
    hits = 0
    dcg = 0.0
    matched_gt_indices = set()
    hit_details = []

    for rank, (r_ticker, r_start, score) in enumerate(retrieved[:top_k], 1):
        r_dt = parse_date(r_start)
        if not r_dt:
            continue
            
        hit_found = False
        for gt_idx, (_, gt_dt) in enumerate(gt_parsed):
            # Date proximity only (no Match_Ticker vs retrieved ticker constraint).
            if abs((r_dt - gt_dt).days) <= proximity_days:
                if gt_idx not in matched_gt_indices:
                    hit_found = True
                    matched_gt_indices.add(gt_idx)
                    break
        
        if hit_found:
            hits += 1
            dcg += 1.0 / np.log2(rank + 1)
            hit_details.append(f"{r_ticker}@{r_start}")

    recall = hits / len(gt_list) if gt_list else 0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_list), top_k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return recall, ndcg, hit_details


# ================================================================
# Sub-step Implementations
# ================================================================
def _eval_dense(conf: Dict):
    """Evaluate dense retrieval using EmbeddingRetriever for the active model."""
    data = conf["data"]
    ev_conf = conf["evaluate"]
    branch = conf["experiment"]["NewsAgg"]
    model_key = conf["experiment"]["EmbModel"]
    
    # Dense corpus: agg/sum use Step 06 parquets (finetuned vs pretrained per evaluate.dense_embedding_source).
    # POOL never runs Step 06; Step 03 writes *_5d_pool.parquet under news_5d_summarized_dir.
    if branch == "pool":
        emb_dir = data["news_5d_summarized_dir"]
    elif ev_conf.get("dense_embedding_source", "finetuned") == "pretrained":
        emb_dir = data.get("embeddings_pretrained_dir") or data["embeddings_dir"]
        logger.info("Dense eval corpus: pretrained (embeddings_pretrained_dir)")
    else:
        emb_dir = data["embeddings_dir"]

    logger.info(f"--- Evaluating Dense Retrieval [{model_key.upper()}] ({branch} branch) ---")
    logger.info(f"Corpus: {emb_dir}")
    
    # Load GT
    start_dt = parse_date(ev_conf["eval_start_date"])
    end_dt = parse_date(ev_conf["eval_end_date"])
    gt_dict = load_dtw_ground_truth(data["dtw_results_dir"], start_dt, end_dt)
    
    # Branch-specific model loading (Experiment 3: POOL)
    vector_model_path = None
    pooling_type = "avg"
    if branch == "pool":
        vector_model_path = os.path.join(conf["finetune"]["output_dir"], "vector_model.pt")
        pooling_type = conf["pool"]["pooling_type"]
        logger.info(f"Using vector projection model: {vector_model_path}")

    # Build Retriever
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dim = conf["models"][model_key]["dim"]
    retriever = EmbeddingRetriever(
        emb_dir, 
        device=device,
        vector_model_path=vector_model_path,
        pooling_type=pooling_type,
        emb_dim=emb_dim
    )
    if not retriever.ready:
        return

    logger.info(
        f"Dense corpus loaded: {retriever.corpus_size} vectors | "
        f"GT queries in eval window: {len(gt_dict)}"
    )

    missing_exact = None
    idx = getattr(retriever, "db_query_dict", None)
    if isinstance(idx, dict) and idx:
        missing_exact = sum(1 for k in gt_dict if k not in idx)
        logger.info(
            f"Query–corpus key check: GT keys without exact (ticker, Window_End) match "
            f"= {missing_exact} / {len(gt_dict)} (retriever snaps to last Window_End on/before Query_Date)"
        )

    if hasattr(retriever, "reset_resolve_stats"):
        retriever.reset_resolve_stats()

    # Run Eval
    recalls, ndcgs = [], []
    n_empty_retrieval = 0
    for (q_ticker, q_date), gt_matches in tqdm(gt_dict.items(), desc=f"Eval {model_key.upper()}"):
        matches = retriever.search(q_ticker, q_date, top_k=50)  # Fetch more for self-hit filtering
        if not matches:
            n_empty_retrieval += 1
        recall, ndcg, _ = compute_metrics(matches, gt_matches, top_k=ev_conf["top_k"], proximity_days=ev_conf["proximity_days"])
        recalls.append(recall)
        ndcgs.append(ndcg)

    rs = getattr(retriever, "_resolve_stats", None)
    if isinstance(rs, dict):
        n_fail = int(rs.get("fail_parse", 0)) + int(rs.get("fail_no_ticker", 0)) + int(rs.get("fail_before_first_window", 0))
        logger.info(
            f"Dense query-key resolution: {rs!r} | unresolved_queries={n_fail}"
        )
        # region agent log
        try:
            _dbg = {
                "sessionId": "f11845",
                "runId": "dense-eval",
                "hypothesisId": "H1",
                "location": "step07_evaluate.py:dense_eval",
                "message": "query_key_resolution_after_snap",
                "data": {
                    "missing_exact_keys": missing_exact,
                    "resolve_stats": dict(rs),
                    "n_gt": len(gt_dict),
                    "n_empty_retrieval": n_empty_retrieval,
                },
                "timestamp": int(time.time() * 1000),
            }
            with open(
                "/mnt/raid1/ken/Capstone_data/.cursor/debug-f11845.log",
                "a",
                encoding="utf-8",
            ) as _f:
                _f.write(json.dumps(_dbg, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # endregion

    if not recalls:
        logger.warning("Dense eval: no queries evaluated (empty GT in date range?).")
        return

    r_arr = np.asarray(recalls, dtype=np.float64)
    n_arr = np.asarray(ndcgs, dtype=np.float64)
    logger.info(
        f"Per-query stats: recall nonzero={int(np.sum(r_arr > 0))}/{len(r_arr)} "
        f"min={np.min(r_arr):.6f} max={np.max(r_arr):.6f}"
    )
    logger.info(
        f"Per-query stats: ndcg nonzero={int(np.sum(n_arr > 0))}/{len(n_arr)} "
        f"min={np.min(n_arr):.6f} max={np.max(n_arr):.6f}"
    )

    logger.info(f"[{model_key.upper()} Results] Recall@10: {np.mean(recalls):.4f} | NDCG@10: {np.mean(ndcgs):.4f}")


def _eval_bm25(conf: Dict):
    """Evaluate lexical retrieval using BM25."""
    data = conf["data"]
    ev_conf = conf["evaluate"]
    corpus_dir = data["news_5d_summarized_dir"]
    
    logger.info("--- Evaluating BM25 Lexical Retrieval ---")
    
    # Load GT
    start_dt = parse_date(ev_conf["eval_start_date"])
    end_dt = parse_date(ev_conf["eval_end_date"])
    gt_dict = load_dtw_ground_truth(data["dtw_results_dir"], start_dt, end_dt)
    
    # Build BM25 Corpus
    logger.info(f"Building BM25 index from {corpus_dir}")
    tokens, meta = [], []
    for f in tqdm(glob.glob(os.path.join(corpus_dir, "*_5d_summaries.csv")), desc="Indexing"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                text = row.get("LLM_5D_Trend_Summary", "")
                s_dt, e_dt = parse_date(row["Window_Start"]), parse_date(row["Window_End"])
                if not s_dt or not e_dt or not isinstance(text, str):
                    continue
                # agg uses lowercase ticker; sum/14B exports often use Ticker
                tick = row.get("ticker")
                if tick is None or (isinstance(tick, float) and pd.isna(tick)):
                    tick = row.get("Ticker")
                if tick is None or (isinstance(tick, float) and pd.isna(tick)):
                    continue
                meta.append((str(tick), s_dt.strftime("%Y-%m-%d"), e_dt.strftime("%Y-%m-%d")))
                tokens.append(tokenize(text))
        except: continue
        
    if not tokens:
        logger.error("BM25 corpus empty.")
        return
        
    bm25 = FastBM25(tokens)
    query_lookup = {(m[0], m[2]): i for i, m in enumerate(meta)}
    db_end_dates = np.array([parse_date(m[2]) for m in meta])
    
    # Run Eval
    recalls, ndcgs = [], []
    for (q_ticker, q_date), gt_matches in tqdm(gt_dict.items(), desc="Eval BM25"):
        q_idx = query_lookup.get((q_ticker, q_date))
        if q_idx is None: continue
        
        scores = bm25.get_scores(tokens[q_idx])
        
        # Temporal mask
        q_dt_obj = parse_date(q_date)
        valid_mask = db_end_dates < q_dt_obj
        scores[~valid_mask] = -float('inf')
        
        # Rank
        top_indices = np.argsort(scores)[::-1][:50]
        retrieved = []
        for idx in top_indices:
            if scores[idx] == -float('inf'): break
            m = meta[idx]
            if m[0] == q_ticker and m[2] == q_date: continue # self
            retrieved.append((m[0], m[1], scores[idx]))
            if len(retrieved) >= 10: break
            
        recall, ndcg, _ = compute_metrics(retrieved, gt_matches, top_k=ev_conf["top_k"], proximity_days=ev_conf["proximity_days"])
        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        logger.warning("BM25: no evaluated queries (check ticker column Ticker vs ticker, or corpus/GT date alignment).")
        return

    logger.info(f"[BM25 Results] Recall@10: {np.mean(recalls):.4f} | NDCG@10: {np.mean(ndcgs):.4f}")


def _eval_cross(conf: Dict):
    """Comparative analysis across all methods."""
    logger.info("--- Cross-Method Error Analysis ---")
    logger.info("Note: Run individual evaluations first to see per-method performance.")
    pass


# ================================================================
# Runner
# ================================================================
_SUBSTEPS = {
    "dense": _eval_dense,
    "active": _eval_dense,
    "bge": _eval_dense,
    "contriever": _eval_dense,
    "bm25": _eval_bm25,
    "cross": _eval_cross,
}


def run(conf: Dict, RetModel: Optional[str] = None):
    """Execute step 07."""
    if RetModel:
        if RetModel not in _SUBSTEPS:
            raise ValueError(f"Unknown RetModel-step '{RetModel}'")
        _SUBSTEPS[RetModel](conf)
    else:
        logger.info("Running standard dense evaluation for active model...")
        _eval_dense(conf)
