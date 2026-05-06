"""
Step 06 — Multi-Branch Embedding Generation
=============================================
Routes embedding generation based on the active branch:
- AGG / SUM: Embed aggregated 5D windows using either the fine-tuned model (default)
  or HuggingFace pretrained weights (`--RetModel base`).
- POOL: Vectors are produced in Step 03; Stage 06 skips (base RetModel-step is a no-op).
"""

import os
import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utility import get_logger, ensure_dir, parse_date

logger = get_logger("step06")


def _run_text_embedding(conf: Dict):
    """(AGG/SUM) Generate embeddings using the Step 05 fine-tuned SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    data = conf["data"]
    branch = conf["experiment"]["NewsAgg"]
    ft_conf = conf["finetune"]

    model_path = ft_conf["output_dir"]
    batch_size = ft_conf["batch_size"]

    input_dir = data["news_5d_summarized_dir"]
    output_dir = ensure_dir(data["embeddings_dir"])

    if not os.path.exists(model_path):
        logger.error(f"Fine-tuned model not found at {model_path}. Run Step 05 first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading fine-tuned model ({branch}) from {model_path}")
    model = SentenceTransformer(
        model_path,
        device=device,
        model_kwargs={"use_safetensors": True},
    )

    _encode_summaries_to_parquet(model, input_dir, output_dir, batch_size, branch)


def _run_text_embedding_pretrained(conf: Dict):
    """(AGG/SUM) Baseline: encode with pretrained model from config (no Step 05 checkpoint)."""
    from sentence_transformers import SentenceTransformer

    data = conf["data"]
    branch = conf["experiment"]["NewsAgg"]
    model_key = conf["experiment"]["EmbModel"]
    ft_conf = conf["finetune"]

    model_name = conf["models"][model_key]["name"]
    batch_size = ft_conf["batch_size"]
    input_dir = data["news_5d_summarized_dir"]
    out_key = "embeddings_pretrained_dir"
    if out_key not in data:
        logger.error(f"config data.{out_key} is missing; add embeddings_pretrained_dir to config.yaml")
        return
    output_dir = ensure_dir(data[out_key])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Loading pretrained {model_key} from {model_name} -> {output_dir} (baseline, no fine-tune)"
    )
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"use_safetensors": True},
    )
    _encode_summaries_to_parquet(model, input_dir, output_dir, batch_size, branch)


def _encode_summaries_to_parquet(model, input_dir: str, output_dir: str, batch_size: int, branch: str):
    csv_files = glob.glob(os.path.join(input_dir, "*_5d_summaries.csv"))
    logger.info(f"Embedding {len(csv_files)} window files for branch: {branch}")

    for fpath in tqdm(csv_files, desc="Embedding windows"):
        out_path = os.path.join(output_dir, os.path.basename(fpath).replace(".csv", ".parquet"))
        if os.path.exists(out_path):
            continue

        try:
            df = pd.read_csv(fpath)
            if df.empty or "LLM_5D_Trend_Summary" not in df.columns:
                continue

            texts = df["LLM_5D_Trend_Summary"].fillna("").astype(str).tolist()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            df["embedding"] = list(embeddings)
            df.rename(columns={"Ticker": "ticker"}, inplace=True)
            df.to_parquet(out_path, index=False)
        except Exception as e:
            logger.warning(f"Error embedding {fpath}: {e}")


def _run_pool_embedding(conf: Dict):
    """(POOL) For Experiment 3, vectors are mostly generated in Step 03."""
    logger.info("POOL Branch: Embeddings for Experiment 3 are pre-calculated in Step 03.")
    logger.info("Skipping Stage 06...")


def run(conf: Dict, RetModel: Optional[str] = None):
    branch = conf["experiment"]["NewsAgg"]
    use_pretrained = RetModel in ("base", "pretrained")

    if branch in ["agg", "sum"]:
        if use_pretrained:
            _run_text_embedding_pretrained(conf)
        else:
            _run_text_embedding(conf)
    else:
        if use_pretrained:
            logger.info(
                "POOL: pretrained dense baseline uses Step 03 parquets; "
                "set evaluate.dense_embedding_source=pretrained and omit vector_model.pt for pure baseline."
            )
        _run_pool_embedding(conf)
