"""
train_metrics.py — Shared training instrumentation
===================================================
Branch/model-agnostic helpers used by both ContrastiveTrainer (text, agg/sum)
and VectorBasedTrainer (vectors, pool):

- `build_log_dir`         : logs/{branch}_{model}/{run_id}/
- `MetricsWriter`         : wandb + CSV double-write, end-of-training plots
- `EpochEvaluator`        : Step 07–style mean Recall@10 / NDCG@10 (DTW GT) each
                            epoch; `query_split: val` (holdout window) or `evaluate`
                            (Step 07 date range); optional `n_queries` cap.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utility import get_logger, load_dtw_ground_truth, parse_date

_dtw_log = get_logger("epoch_dtw_eval")


def build_log_dir(conf: Dict, run_id: Optional[str] = None) -> str:
    """Return logs/{branch}_{model}/{run_id}/ under project_root."""
    branch = conf["experiment"]["NewsAgg"]
    model = conf["experiment"]["EmbModel"]
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = conf.get("project_root") or os.getcwd()
    log_dir = os.path.join(root, "logs", f"{branch}_{model}", run_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def sanitize_metric_key(name: str, prefix: str = "ir_") -> str:
    """Make a single CSV / wandb key from InformationRetrieval-style names."""
    safe = re.RetModel(r"[^0-9a-zA-Z@._-]+", "_", str(name).strip())
    return prefix + safe.replace("@", "at").replace(".", "_")


def flatten_ir_metrics_to_row(ir_out: Any) -> Dict[str, float]:
    """ST IR evaluator returns dict[str, float]; add ir_* keys for wide CSV."""
    if not isinstance(ir_out, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in ir_out.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            out[sanitize_metric_key(k)] = float(v)
    return out


class MetricsWriter:
    """CSV + wandb double-writer with end-of-training plots. Uses a wide, dynamic
    schema so each row can add `val_recall@10` / `ir_*` etc. without fixed columns.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")

    def log(self, metrics: Dict, use_wandb: bool = True) -> None:
        """Append one row; union column set across rows; mirror numeric payload to wandb."""
        row: Dict = {}
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, (np.floating, float)):
                row[k] = float(v)
            elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                row[k] = int(v)
            elif isinstance(v, (str, bool)):
                if isinstance(v, bool):
                    continue
                row[k] = str(v)[:2000]
        if not row:
            return
        new_df = pd.DataFrame([row])
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                old = pd.read_csv(self.csv_path)
            except Exception:
                old = pd.DataFrame()
            df = pd.concat([old, new_df], ignore_index=True, sort=False)
        else:
            df = new_df
        cols = list(df.columns)
        if "epoch" in cols:
            ordered = ["epoch"] + sorted([c for c in cols if c != "epoch"], key=str)
        else:
            ordered = sorted(cols, key=str)
        df = df.reindex(columns=ordered)
        df.to_csv(self.csv_path, index=False)

        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    payload = {k: v for k, v in row.items() if v is not None}
                    step = payload.get("epoch")
                    if step is not None:
                        wandb.log(payload, step=int(step))
                    else:
                        wandb.log(payload)
            except Exception:
                pass

    def plot(self) -> None:
        """Render loss_curves.png and retrieval/val_metrics_curves.png from metrics.csv."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) < 2:
                return
            df = pd.read_csv(self.csv_path)
            if df.empty or "epoch" not in df.columns:
                return

            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for col, style in (("train_loss", "o-"), ("val_loss", "s--")):
                if col in df.columns and df[col].notna().any():
                    ax.plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), style, label=col)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, "loss_curves.png"), dpi=120)
            plt.close(fig)

            # Second figure: Step07-style val_* first, then legacy names
            score_cols: List[Tuple[str, str]] = []
            for c in ("val_recall@10", "val_ndcg@10"):
                if c in df.columns and df[c].notna().any():
                    score_cols.append((c, "o-" if "recall" in c else "s--"))
            if not score_cols:
                for c in df.columns:
                    c = str(c)
                    if c in ("epoch", "train_loss", "val_loss", "lr", "temperature", "val_n", "subset_n"):
                        continue
                    if c in ("val_ir_score",):
                        score_cols.append((c, "s--"))
                    elif c.startswith("ir_") and "ndcg" in c.lower():
                        score_cols.append((c, "o-"))
            if not score_cols:
                for c in ("subset_recall@10", "subset_ndcg@10"):
                    if c in df.columns and df[c].notna().any():
                        score_cols.append((c, "o-" if "recall" in c else "s--"))
            if not score_cols:
                for c in df.columns:
                    c = str(c)
                    if c.startswith("ir_") and c not in (x[0] for x in score_cols):
                        score_cols.append((c, "o-"))
                    if len(score_cols) >= 4:
                        break
            if score_cols:
                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                for c, st in score_cols[:6]:
                    if c in df.columns and df[c].notna().any():
                        ax.plot(
                            df["epoch"],
                            pd.to_numeric(df[c], errors="coerce"),
                            st,
                            label=c[:40],
                        )
                ax.set_xlabel("epoch")
                ax.set_ylabel("metric")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
                fig.tight_layout()
                vpath = os.path.join(self.log_dir, "val_metrics_curves.png")
                fig.savefig(vpath, dpi=120)
                try:
                    import shutil
                    shutil.copy2(vpath, os.path.join(self.log_dir, "retrieval_curves.png"))
                except Exception:
                    pass
                plt.close(fig)
        except Exception as e:
            print(f"[MetricsWriter.plot] skipped: {e}")


# ----------------------------------------------------------------------
# EpochEvaluator: Step 07–style mean Recall@10 / NDCG@10 (per epoch)
# ----------------------------------------------------------------------


def _compute_one_query(
    retrieved: List[Tuple[str, str, float]],
    gt_matches: List[Tuple[str, str]],
    top_k: int,
    proximity_days: int,
) -> Tuple[float, float]:
    """Mirror of pipeline.step07_evaluate.compute_metrics (recall, ndcg)."""
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


def build_dtw_epoch_evaluator(
    conf: Dict, device: Optional[str] = None
) -> Optional["EpochEvaluator"]:
    """Return EpochEvaluator if enabled in finetune.dtw_epoch_eval (or legacy subset_eval)."""
    finetune = conf.get("finetune", {})
    dtw = finetune.get("dtw_epoch_eval")
    legacy = finetune.get("subset_eval", {})
    if isinstance(dtw, dict) and dtw.get("enabled", True):
        return EpochEvaluator(conf, device=device, _legacy_mode=False)
    if legacy.get("enabled", False):
        return EpochEvaluator(conf, device=device, _legacy_mode=True, _legacy=legacy)
    return None


class EpochEvaluator:
    """Step 07–aligned mean Recall@10 / NDCG@10 on DTW ground truth, each epoch.

    Branch-aware: encodes 5D text/vector corpora and scores like
    `EmbeddingRetriever` + `step07_evaluate.compute_metrics`.

    `query_split` (finetune.dtw_epoch_eval):
    - `val` — same window as step04 `val` rows
      (val_start ≤ Query_Date < val_end_exclusive)
    - `evaluate` — same as Step 07 (`evaluate.eval_start_date` … `eval_end_date`)

    `n_queries`: null/omitted = all queries in that split; positive int = stratified cap.
    """

    def __init__(
        self,
        conf: Dict,
        n_queries: int = 80,
        seed: int = 42,
        device: Optional[str] = None,
        _legacy_mode: bool = False,
        _legacy: Optional[Dict] = None,
    ) -> None:
        self.conf = conf
        self.branch = conf["experiment"]["NewsAgg"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _legacy = _legacy or {}

        ev = conf["evaluate"]
        self.top_k = int(ev.get("top_k", 10))
        self.proximity_days = int(ev.get("proximity_days", 5))
        dtw_dir = conf["data"]["dtw_results_dir"]
        dcfg = conf.get("finetune", {}).get("dtw_epoch_eval") or {}

        if _legacy_mode:
            self._seed = int(_legacy.get("seed", 42))
            self.query_split = "evaluate"
            n_cap = _legacy.get("n_queries", n_queries)
            self.n_queries_cap: Optional[int] = int(n_cap) if n_cap is not None else 80
            start = parse_date(ev["eval_start_date"])
            end = parse_date(ev["eval_end_date"])
            full_gt = load_dtw_ground_truth(dtw_dir, start, end)
        else:
            self._seed = int(dcfg.get("seed", seed))
            self.query_split = str(dcfg.get("query_split", "val")).lower().strip()
            nq = dcfg.get("n_queries", None)
            if nq is None or (isinstance(nq, str) and str(nq).lower() in ("", "null", "none")):
                self.n_queries_cap = None
            else:
                self.n_queries_cap = max(1, int(nq))

            if self.query_split == "val":
                vs = parse_date(
                    dcfg.get("val_start", "2025-01-29")
                )  # match pipeline.step04
                vex = parse_date(
                    dcfg.get("val_end_exclusive", "2025-04-10")
                )
                if not vs or not vex:
                    full_gt = {}
                else:
                    end_incl = vex - timedelta(days=1)
                    full_gt = load_dtw_ground_truth(dtw_dir, vs, end_incl)
            elif self.query_split == "evaluate":
                start = parse_date(ev["eval_start_date"])
                end = parse_date(ev["eval_end_date"])
                full_gt = load_dtw_ground_truth(dtw_dir, start, end)
            else:
                raise ValueError(
                    f"finetune.dtw_epoch_eval.query_split must be 'val' or 'evaluate', got {self.query_split!r}"
                )

        n_cap = self.n_queries_cap
        if n_cap is not None and len(full_gt) > n_cap:
            self.eval_gt = self._stratified_sample(full_gt, n_cap, self._seed)
        else:
            self.eval_gt = dict(full_gt)

        self.eval_tickers = sorted({k[0] for k in self.eval_gt})

        if self.branch == "pool":
            self._load_pool_corpus()
        else:
            self._load_text_corpus()

        self._end_dts_np = np.array(
            [parse_date(m[2]) for m in self.meta], dtype=object
        )

        self._by_ticker: Dict[str, List[Tuple[datetime, str, int]]] = defaultdict(list)
        for i, (t, _, e) in enumerate(self.meta):
            dt = parse_date(e)
            if dt:
                self._by_ticker[t].append((dt, e, i))
        for t in self._by_ticker:
            self._by_ticker[t].sort(key=lambda x: x[0])

        _ft_dcfg = conf.get("finetune", {}).get("dtw_epoch_eval") or {}
        self.encode_batch_size = max(1, int(_ft_dcfg.get("encode_batch_size", 64)))

    @staticmethod
    def _stratified_sample(
        full_gt: Dict[Tuple[str, str], List[Tuple[str, str]]],
        n_queries: int,
        seed: int,
    ) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        if not full_gt or n_queries <= 0:
            return {}
        rng = np.random.default_rng(seed)
        by_t: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for k in full_gt:
            by_t[k[0]].append(k)
        for t in by_t:
            by_t[t] = list(by_t[t])
            rng.shuffle(by_t[t])
        tickers = sorted(by_t.keys())
        rng.shuffle(tickers)
        picked: List[Tuple[str, str]] = []
        ti = 0
        while len(picked) < n_queries and tickers:
            t = tickers[ti % len(tickers)]
            if by_t[t]:
                picked.append(by_t[t].pop())
                ti += 1
            else:
                tickers.remove(t)
        return {k: full_gt[k] for k in picked}

    # ------------------------------------------------------------------
    # Corpus loaders (called once)
    # ------------------------------------------------------------------
    def _load_pool_corpus(self) -> None:
        news_dir = self.conf["data"]["news_5d_summarized_dir"]
        self.meta: List[Tuple[str, str, str]] = []
        self.raw_embeddings: List[np.ndarray] = []
        self.is_seq = False
        for t in self.eval_tickers:
            pq = os.path.join(news_dir, f"{t}_5d_pool.parquet")
            if not os.path.exists(pq):
                continue
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue
            emb_col = "embedding_seq" if "embedding_seq" in df.columns else "embedding"
            if emb_col not in df.columns or df.empty:
                continue
            self.is_seq = self.is_seq or (emb_col == "embedding_seq")
            for _, row in df.iterrows():
                s = parse_date(row.get("Window_Start"))
                e = parse_date(row.get("Window_End"))
                if not s or not e:
                    continue
                self.meta.append(
                    (row["ticker"], s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
                )
                self.raw_embeddings.append(np.asarray(row[emb_col]))

    def _load_text_corpus(self) -> None:
        news_dir = self.conf["data"]["news_5d_summarized_dir"]
        self.meta = []
        self.texts: List[str] = []
        for t in self.eval_tickers:
            csv_path = os.path.join(news_dir, f"{t}_5d_summaries.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            text_col = None
            for c in ("LLM_5D_Trend_Summary", "summary", "text"):
                if c in df.columns:
                    text_col = c
                    break
            if text_col is None:
                continue
            for _, row in df.iterrows():
                s = parse_date(row.get("Window_Start"))
                e = parse_date(row.get("Window_End"))
                if not s or not e:
                    continue
                text = row.get(text_col)
                if not isinstance(text, str) or len(text) < 5:
                    continue
                t_col = row.get("ticker", row.get("Ticker", t))
                self.meta.append(
                    (t_col, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
                )
                self.texts.append(text)

    # ------------------------------------------------------------------
    # Encoders (called per epoch)
    # ------------------------------------------------------------------
    def _encode_pool(self, model) -> torch.Tensor:
        import torch.nn.functional as F

        was_training = model.training
        model.eval()
        vecs: List[torch.Tensor] = []
        with torch.no_grad():
            for raw in tqdm(
                self.raw_embeddings,
                total=len(self.raw_embeddings),
                desc="DTW corpus encode (pool)",
                unit="row",
                leave=True,
                file=sys.stderr,
                mininterval=0.2,
            ):
                arr = np.asarray(raw)
                t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                elif t.dim() == 2:
                    t = t.unsqueeze(0)
                # Reuse the model's own projector path for queries
                q, _, _ = model(t, t)
                vecs.append(q.squeeze(0).float().cpu())
        if was_training:
            model.train()
        db = torch.stack(vecs, dim=0)
        db = F.normalize(db, p=2, dim=1)
        return db

    def _encode_text(self, model) -> torch.Tensor:
        was_training = getattr(model, "training", False)
        try:
            model.eval()
        except Exception:
            pass
        texts = self.texts
        bs = self.encode_batch_size
        parts: List[torch.Tensor] = []
        n = len(texts)
        n_batch = max(1, (n + bs - 1) // bs)
        with torch.no_grad():
            # Batched model.encode: tqdm shows batch k / n_batch (if not TTY, set TERM or use epoch_dtw_log).
            for i in tqdm(
                range(0, n, bs),
                total=n_batch,
                desc="DTW corpus encode",
                unit="batch",
                leave=True,
                file=sys.stderr,
                mininterval=0.2,
            ):
                chunk = texts[i : i + bs]
                e = model.encode(
                    chunk,
                    batch_size=max(1, len(chunk)),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                parts.append(e.float().cpu())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        if was_training:
            try:
                model.train()
            except Exception:
                pass
        if not parts:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, model) -> Dict:
        if not self.meta or not self.eval_gt:
            return {"val_recall@10": 0.0, "val_ndcg@10": 0.0, "val_n": 0}

        n_texts = len(self.texts) if self.branch != "pool" else len(self.raw_embeddings)
        n_q = len(self.eval_gt)
        _dtw_log.info(
            f"DTW eval start: {n_q} queries, corpus {n_texts} rows, branch={self.branch!r}. "
            f"Encoding the full corpus each time dominates wall time (often 10–90+ min on GPU for agg/sum)."
        )
        t0 = time.perf_counter()
        db = self._encode_pool(model) if self.branch == "pool" else self._encode_text(model)
        _dtw_log.info(
            f"DTW eval corpus encode done in {time.perf_counter() - t0:.1f}s; scoring {n_q} queries…"
        )
        db_np = db.numpy()
        end_dts = self._end_dts_np

        recalls: List[float] = []
        ndcgs: List[float] = []

        for (q_t, q_date), gt_matches in tqdm(
            self.eval_gt.items(),
            total=len(self.eval_gt),
            desc="DTW val queries",
            unit="q",
            leave=True,
            file=sys.stderr,
            mininterval=0.2,
        ):
            q_dt = parse_date(q_date)
            if q_dt is None:
                continue
            cands = self._by_ticker.get(q_t, [])
            if not cands:
                continue
            idx = None
            for d, _, i in reversed(cands):
                if d <= q_dt:
                    idx = i
                    break
            if idx is None:
                continue

            q_vec = db_np[idx]
            scores = db_np @ q_vec
            mask = np.array([d is not None and d < q_dt for d in end_dts])
            scores = np.where(mask, scores, -np.inf)
            resolved_end = self.meta[idx][2]
            for j, (mt, _, me) in enumerate(self.meta):
                if mt == q_t and me == resolved_end:
                    scores[j] = -np.inf

            order = np.argsort(-scores)[: max(50, self.top_k * 5)]
            retrieved: List[Tuple[str, str, float]] = []
            for j in order:
                s = float(scores[j])
                if not np.isfinite(s):
                    break
                m = self.meta[j]
                retrieved.append((m[0], m[1], s))
                if len(retrieved) >= max(50, self.top_k * 5):
                    break

            r, n = _compute_one_query(
                retrieved, gt_matches, self.top_k, self.proximity_days
            )
            recalls.append(r)
            ndcgs.append(n)

        _dtw_log.info(
            f"DTW eval finished in {time.perf_counter() - t0:.1f}s total "
            f"(val_recall@10={float(np.mean(recalls)) if recalls else 0.0:.4f}, "
            f"val_n={len(recalls)})"
        )
        return {
            "val_recall@10": float(np.mean(recalls)) if recalls else 0.0,
            "val_ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "val_n": len(recalls),
        }
