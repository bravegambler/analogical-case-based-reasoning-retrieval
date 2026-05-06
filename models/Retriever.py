"""
Retriever — DTW & Embedding-based Retrieval
=============================================
Core retrieval logic for evaluation:
- Support for both standard BGE-M3 text embeddings and
  branch-specific vector transformations (Experiment 3).
"""

import os
import glob
import bisect
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utility import parse_date


class EmbeddingRetriever:
    """Vector-similarity retrieval over a pre-embedded corpus."""

    def __init__(self, emb_dir: str, device: str = "cuda", vector_model_path: str = None, pooling_type: str = "avg", emb_dim: int = 1024):
        self.device = device
        self.db_tensor: Optional[torch.Tensor] = None
        self.db_meta: Optional[List[Tuple[str, str, str]]] = None 
        self.db_end_dates: Optional[np.ndarray] = None
        self.db_query_dict: Dict[Tuple[str, str], int] = {}
        self._ends_by_ticker: Dict[str, List[Tuple[datetime, str]]] = {}
        self._end_dts_by_ticker: Dict[str, List[datetime]] = {}
        self._resolve_stats: Dict[str, int] = {}

        # For Experiment 3: Load the projection/pooling model
        self.vector_model = None
        if vector_model_path and os.path.exists(vector_model_path):
            from models.VectorBasedTrainer import VectorContrastiveModel
            self.vector_model = VectorContrastiveModel(emb_dim=emb_dim, pooling_type=pooling_type).to(device)
            self.vector_model.load_state_dict(torch.load(vector_model_path, map_location=device))
            self.vector_model.eval()
            print(f"✅ Loaded vector model from {vector_model_path} (dim={emb_dim})")

        self._build_corpus(emb_dir)

    def _build_corpus(self, emb_dir: str):
        """Load parquet embeddings into a GPU-resident normalized tensor."""
        embeddings_list = []
        metadata = []

        pq_files = glob.glob(os.path.join(emb_dir, "*.parquet"))
        if not pq_files:
            print(f"❌ No embeddings found in {emb_dir}")
            return

        for pq_file in tqdm(pq_files, desc="Building corpus"):
            try:
                df = pd.read_parquet(pq_file)
                # Check for either attention seq or pooled vector
                emb_col = "embedding_seq" if "embedding_seq" in df.columns else "embedding"
                if emb_col not in df.columns or df.empty:
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
                    embeddings_list.append(row[emb_col])
            except Exception:
                continue

        if not embeddings_list:
            return

        # Move to GPU
        with torch.no_grad():
            if self.vector_model:
                # Transform each embedding/sequence through the fine-tuned vector model
                final_embs = []
                for item in embeddings_list:
                    tensor = torch.tensor(np.array(item), dtype=torch.float32).to(self.device).unsqueeze(0)
                    # VectorContrastiveModel handles both single vector and sequence
                    q_proj, _, _ = self.vector_model(tensor, tensor) # dummy p
                    final_embs.append(q_proj.cpu())
                raw_tensor = torch.cat(final_embs, dim=0)
            else:
                raw_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)

        self.db_tensor = F.normalize(raw_tensor, p=2, dim=1).to(self.device)
        self.db_meta = metadata
        self.db_query_dict = {(m[0], m[2]): idx for idx, m in enumerate(metadata)}
        self.db_end_dates = np.array([parse_date(m[2]) for m in metadata])

        by_t: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        for m in metadata:
            ed = parse_date(m[2])
            if ed:
                by_t[m[0]].append((ed, m[2]))
        for t in by_t:
            by_t[t].sort(key=lambda x: x[0])
        self._ends_by_ticker = dict(by_t)
        self._end_dts_by_ticker = {t: [p[0] for p in pairs] for t, pairs in self._ends_by_ticker.items()}
        self.reset_resolve_stats()

        print(f"✅ Corpus ready: {len(metadata)} vectors.")

    @property
    def ready(self) -> bool:
        return self.db_tensor is not None

    @property
    def corpus_size(self) -> int:
        return len(self.db_meta) if self.db_meta else 0

    def reset_resolve_stats(self) -> None:
        """Reset per-eval counters for query-key resolution (exact vs snapped vs fail)."""
        self._resolve_stats = {
            "exact": 0,
            "snapped": 0,
            "fail_parse": 0,
            "fail_no_ticker": 0,
            "fail_before_first_window": 0,
        }

    def _resolve_query_key(self, query_ticker: str, query_date: str) -> Optional[Tuple[str, str]]:
        """Map GT (ticker, query_date) to a corpus row key (ticker, Window_End).

        Pool windows only exist on trading days; DTW Query_Date may be a non-trading day.
        Use the latest corpus Window_End on or before query_date for that ticker.
        """
        k = (query_ticker, query_date)
        if k in self.db_query_dict:
            self._resolve_stats["exact"] += 1
            return k
        q_dt = parse_date(query_date)
        if not q_dt:
            self._resolve_stats["fail_parse"] += 1
            return None
        dts = self._end_dts_by_ticker.get(query_ticker)
        if not dts:
            self._resolve_stats["fail_no_ticker"] += 1
            return None
        i = bisect.bisect_right(dts, q_dt) - 1
        if i < 0:
            self._resolve_stats["fail_before_first_window"] += 1
            return None
        end_str = self._ends_by_ticker[query_ticker][i][1]
        resolved = (query_ticker, end_str)
        if resolved not in self.db_query_dict:
            self._resolve_stats["fail_before_first_window"] += 1
            return None
        self._resolve_stats["snapped"] += 1
        return resolved

    def search(self, query_ticker: str, query_date: str, top_k: int = 10, exclude_self: bool = True) -> List[Tuple[str, str, float]]:
        if not self.ready: return []
        resolved = self._resolve_query_key(query_ticker, query_date)
        if resolved is None:
            return []

        query_idx = self.db_query_dict[resolved]
        q_tensor = self.db_tensor[query_idx].unsqueeze(0)
        scores = torch.mm(q_tensor, self.db_tensor.T).squeeze(0)

        # Future leakage filter (use original query calendar date, not snapped Window_End)
        q_date_obj = parse_date(query_date)
        if q_date_obj:
            valid_mask = self.db_end_dates < q_date_obj
            valid_tensor = torch.from_numpy(valid_mask).to(self.device)
            scores[~valid_tensor] = -float("inf")

        top_scores, top_indices = torch.topk(scores, k=min(top_k * 5, len(scores)))
        results = []
        resolved_end = resolved[1]
        for idx in top_indices.cpu().tolist():
            if scores[idx] == -float("inf"): break
            m_ticker, m_start, m_end = self.db_meta[idx]
            if exclude_self and m_ticker == query_ticker and m_end == resolved_end: continue
            results.append((m_ticker, m_start, scores[idx].item()))
            if len(results) >= top_k: break
        return results
