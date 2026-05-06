"""
models/VectorBasedTrainer.py — Custom Contrastive Trainer for Pooled Vectors
============================================================================
Implements a PyTorch training loop for Experiment 3 (POOL).
- Supports Max, Avg, and Attention pooling.
- Uses InfoNCE loss with learnable temperature.
- Optimized for pre-computed BGE/Contriever embeddings.

Per-epoch instrumentation:
- train_loss, val_loss (InfoNCE on val parquet)
- Step 07–style `val_recall@10` / `val_ndcg@10` on DTW ground truth (finetune.dtw_epoch_eval)
- Double-write to wandb and logs/{branch}_{model}/{run_id}/metrics.csv
- End-of-training loss_curves.png and retrieval_curves.png
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb

from utility import get_logger
from .Poolers import AttentionPooler, IdentityPooler
from .train_metrics import MetricsWriter, build_dtw_epoch_evaluator, build_log_dir
from .training_heartbeat import (
    start_training_heartbeat,
    training_header,
    write_run_status,
)

logger = get_logger("vector_trainer")


class VectorDataset(Dataset):
    """Loads pre-computed vector pairs from Parquet."""

    def __init__(self, path: str, is_attention: bool = False):
        self.df = pd.read_parquet(path)
        self.is_attention = is_attention

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_attention:
            q = torch.tensor(row["query_seq"], dtype=torch.float32)
            p = torch.tensor(row["pos_seq"], dtype=torch.float32)
        else:
            q = torch.tensor(row["query_vec"], dtype=torch.float32)
            p = torch.tensor(row["pos_vec"], dtype=torch.float32)
        return q, p


class VectorContrastiveModel(nn.Module):
    """Wrapper that applies pooling and optional projection."""

    def __init__(self, emb_dim: int, pooling_type: str = "attention"):
        super().__init__()
        if pooling_type == "attention":
            self.pooler = AttentionPooler(emb_dim)
        else:
            self.pooler = IdentityPooler(mode=pooling_type)

        self.projector = nn.Linear(emb_dim, emb_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, q_raw, p_raw):
        q = self.pooler(q_raw) if q_raw.dim() == 3 else q_raw
        p = self.pooler(p_raw) if p_raw.dim() == 3 else p_raw

        q = self.projector(q)
        p = self.projector(p)

        q = F.normalize(q, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=-1)

        return q, p, self.logit_scale.exp()


def info_nce_loss(q, p, t):
    """InfoNCE Loss: alignment(q, p) vs alignment(q, all_negatives)."""
    logits = torch.matmul(q, p.T) * t
    labels = torch.arange(len(q)).to(q.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


class VectorBasedTrainer:
    def __init__(self, conf: Dict):
        self.conf = conf
        self.p_conf = conf["pool"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[VectorContrastiveModel] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def _init_model(self, emb_dim: int):
        """Initialize model once dimension is known."""
        self.model = VectorContrastiveModel(
            emb_dim=emb_dim,
            pooling_type=self.p_conf["pooling_type"],
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.p_conf["learning_rate"]),
            weight_decay=0.01,
        )
        logger.info(f"Initialized VectorContrastiveModel with dimension: {emb_dim}")

    def _val_loss(self, val_loader: Optional[DataLoader]) -> Optional[float]:
        """One forward pass over val pairs; returns mean InfoNCE loss."""
        if val_loader is None or self.model is None:
            return None
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for q_raw, p_raw in val_loader:
                q_raw = q_raw.to(self.device)
                p_raw = p_raw.to(self.device)
                q, p, temp = self.model(q_raw, p_raw)
                loss = info_nce_loss(q, p, temp)
                total += float(loss.item())
                count += 1
        self.model.train()
        return total / count if count else None

    def train(self):
        train_path = self.conf["data"]["train_jsonl"].replace(".jsonl", ".parquet")
        val_path = self.conf["data"]["val_jsonl"].replace(".jsonl", ".parquet")
        is_attn = self.p_conf["pooling_type"] == "attention"

        train_ds = VectorDataset(train_path, is_attention=is_attn)
        if len(train_ds) == 0:
            logger.error(f"No training rows in {train_path}; aborting.")
            return

        train_loader = DataLoader(
            train_ds,
            batch_size=self.p_conf["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        val_loader: Optional[DataLoader] = None
        if os.path.exists(val_path):
            try:
                val_ds = VectorDataset(val_path, is_attention=is_attn)
                if len(val_ds) > 0:
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=self.p_conf["batch_size"],
                        shuffle=False,
                        num_workers=0,
                    )
                    logger.info(f"Val pairs loaded: {len(val_ds)} from {val_path}")
            except Exception as e:
                logger.warning(f"Could not load val parquet {val_path}: {e}")

        log_dir = build_log_dir(self.conf)
        writer = MetricsWriter(log_dir)

        run_name = (
            f"pool-{self.conf['experiment']['EmbModel']}-"
            f"{self.p_conf['pooling_type']}"
        )
        wandb.init(
            project="nasdaq100-vector-finetuning",
            name=run_name,
            config={
                "branch": self.conf["experiment"]["NewsAgg"],
                "model": self.conf["experiment"]["EmbModel"],
                "pooling_type": self.p_conf["pooling_type"],
                "batch_size": self.p_conf["batch_size"],
                "learning_rate": self.p_conf["learning_rate"],
                "epochs": self.conf["finetune"]["epochs"],
            },
            dir=log_dir,
            reinit=True,
        )

        try:
            dtw_eval = build_dtw_epoch_evaluator(
                self.conf, device=str(self.device)
            )
        except Exception as e:
            logger.warning(f"DTW epoch evaluator disabled (init failed): {e}")
            dtw_eval = None
        if dtw_eval is not None:
            logger.info(
                f"Step07-style epoch eval: split={getattr(dtw_eval, 'query_split', '?')} "
                f"{len(getattr(dtw_eval, 'eval_gt', {}))} queries, "
                f"corpus rows={len(getattr(dtw_eval, 'meta', []))}"
            )

        _dcfg = self.conf.get("finetune", {}).get("dtw_epoch_eval") or {}
        dtw_eval_every = max(1, int(_dcfg.get("eval_every", 1)))

        n_epochs = int(self.conf["finetune"]["epochs"])
        n_steps_per_epoch = max(1, len(train_loader))
        status_path = os.path.join(log_dir, "run_status.txt")
        shared: Dict = {"global_step": 0, "epoch": 0}
        training_header(status_path, n_steps_per_epoch * n_epochs, n_epochs)
        finet = self.conf.get("finetune") or {}
        hb_sec = int(finet.get("heartbeat_sec", 30))
        stop_hb = start_training_heartbeat(
            status_path,
            hb_sec,
            shared,
            n_steps_per_epoch * n_epochs,
            n_epochs,
            label="pool train+dtw",
        )
        try:
            for epoch in range(n_epochs):
                shared["epoch"] = epoch + 1
                total_loss = 0.0
                n_batches = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
                for q_raw, p_raw in pbar:
                    q_raw = q_raw.to(self.device)
                    p_raw = p_raw.to(self.device)

                    if self.model is None:
                        emb_dim = q_raw.shape[-1]
                        self._init_model(emb_dim)
                        self.model.train()

                    self.optimizer.zero_grad()
                    q, p, temp = self.model(q_raw, p_raw)
                    loss = info_nce_loss(q, p, temp)

                    loss.backward()
                    self.optimizer.step()
                    shared["global_step"] = int(shared.get("global_step", 0)) + 1

                    total_loss += float(loss.item())
                    n_batches += 1
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "temp": f"{1/temp.item():.3f}"}
                    )

                train_loss = total_loss / max(1, n_batches)

                val_loss = self._val_loss(val_loader)

                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6) if val_loss is not None else None,
                    "lr": float(self.p_conf["learning_rate"]),
                    "temperature": float(1.0 / self.model.logit_scale.exp().item()),
                }

                if dtw_eval is not None and self.model is not None:
                    do_dtw = dtw_eval_every <= 1 or ((epoch + 1) % dtw_eval_every == 0)
                    if do_dtw:
                        try:
                            logger.info(
                                "DTW pool eval (full corpus encode; may take a long time — see tqdm/dtw logs)."
                            )
                            metrics.update(dtw_eval.run(self.model))
                        except Exception as e:
                            logger.warning(f"DTW epoch eval failed: {e}")
                    else:
                        logger.info(
                            f"Skipping DTW eval (dtw_epoch_eval.eval_every={dtw_eval_every})."
                        )

                writer.log(metrics)
                logger.info(
                    f"Epoch {epoch+1} | train_loss={train_loss:.4f} "
                    f"val_loss={val_loss if val_loss is None else round(val_loss,4)} "
                    f"val_recall@10={metrics.get('val_recall@10')} "
                    f"val_ndcg@10={metrics.get('val_ndcg@10')}"
                )
                write_run_status(
                    status_path,
                    f"pool epoch {epoch+1}/{n_epochs} done  train_loss={train_loss:.6f}",
                )
        finally:
            stop_hb()
            write_run_status(
                status_path, "==== pool train finished (or interrupted) ====", also_print=True
            )

        writer.plot()

        if self.model is None:
            logger.error("Model was never initialized (empty training loop); nothing to save.")
            wandb.finish()
            return

        os.makedirs(self.conf["finetune"]["output_dir"], exist_ok=True)
        ckpt_path = os.path.join(
            self.conf["finetune"]["output_dir"], "vector_model.pt"
        )
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info(f"Vector model saved to {ckpt_path}")
        logger.info(f"Metrics written to {log_dir}")
        wandb.finish()
