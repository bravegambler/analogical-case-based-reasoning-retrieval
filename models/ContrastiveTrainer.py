"""
ContrastiveTrainer — BGE/Contriever Fine-tuning Wrapper
========================================================
Wraps SentenceTransformer contrastive training with:
  - NoDuplicatesDataLoader for in-batch negatives
  - MultipleNegativesRankingLoss (MNRL)
  - Per-epoch val_loss (MNRL on val pairs) and Step 07–style
    `val_recall@10` / `val_ndcg@10` on DTW ground truth (finetune.dtw_epoch_eval)
  - double-write to wandb and metrics.csv; loss_curves.png and retrieval_curves.png
  - Gradient checkpointing for VRAM efficiency
"""

import os
import sys
import json
from typing import Dict, List, Optional

import torch
import wandb
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    datasets,
)
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device

from utility import get_logger
from .train_metrics import MetricsWriter, build_dtw_epoch_evaluator, build_log_dir
from .training_heartbeat import (
    make_trainer_step_callbacks,
    start_training_heartbeat,
    training_header,
    write_run_status,
)

logger = get_logger("contrastive_trainer")


class _LossTracker:
    """Shared mutable state so the forward hook and evaluator can talk."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def pop_mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        m = self.sum / self.count
        self.sum = 0.0
        self.count = 0
        return m


class _EpochEvalWrapper(SentenceEvaluator):
    """Each epoch: train/val loss and Step 07–style DTW `val_recall@10` / `val_ndcg@10`."""

    def __init__(
        self,
        conf: Dict,
        val_examples: List[InputExample],
        train_loss_module: torch.nn.Module,
        loss_tracker: _LossTracker,
        writer: MetricsWriter,
        epoch_evaluator,
        batch_size: int,
    ):
        super().__init__()
        dcfg = conf.get("finetune", {}).get("dtw_epoch_eval") or {}
        self.dtw_eval_every = max(1, int(dcfg.get("eval_every", 1)))
        self.val_examples = val_examples
        self.loss_module = train_loss_module
        self.tracker = loss_tracker
        self.writer = writer
        self.epoch_evaluator = epoch_evaluator
        self.batch_size = batch_size
        self._epoch_counter = 0
        # SaveModelCallback uses getattr(evaluator, "primary_metric", "evaluator"); BaseEvaluator
        # sets primary_metric=None, so the default is never used — must set a real suffix.
        self.primary_metric = "evaluator"

    def _compute_val_loss(self, model: SentenceTransformer) -> Optional[float]:
        if not self.val_examples:
            return None
        device = model.device
        was_training = self.loss_module.training
        self.loss_module.eval()
        model.eval()
        total = 0.0
        count = 0
        try:
            with torch.no_grad():
                for i in range(0, len(self.val_examples), self.batch_size):
                    batch = self.val_examples[i : i + self.batch_size]
                    if len(batch) < 2:
                        continue
                    features, labels = model.smart_batching_collate(batch)
                    # Only move tensors; preprocess may attach non-Tensor metadata (str, etc.).
                    features = [batch_to_device(feat, device) for feat in features]
                    labels = labels.to(device) if labels is not None else labels
                    loss_val = self.loss_module(features, labels)
                    total += float(loss_val.detach().item())
                    count += 1
        except Exception as e:
            logger.warning(f"val_loss computation failed: {e}")
            total, count = 0.0, 0
        finally:
            if was_training:
                self.loss_module.train()
            model.train()
        return total / count if count else None

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        # SentenceTransformerTrainer passes TrainerState.epoch (float), e.g. 1.0 after
        # the first epoch — NOT a 0-based index. Do not use epoch+1; use a simple counter.
        self._epoch_counter += 1
        epoch_idx = self._epoch_counter

        train_loss = self.tracker.pop_mean()
        val_loss = self._compute_val_loss(model)

        metrics: Dict = {
            "epoch": epoch_idx,
            "train_loss": round(train_loss, 6) if train_loss is not None else None,
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
        }

        if self.epoch_evaluator is not None:
            do_dtw = self.dtw_eval_every <= 1 or (
                epoch_idx % self.dtw_eval_every == 0
            )
            if do_dtw:
                try:
                    logger.info(
                        "End of epoch: running DTW retrieval eval (see tqdm + epoch_dtw_eval logs). "
                        "Full-corpus BGE encode is slow; 30–120 min first time is not unusual."
                    )
                    metrics.update(self.epoch_evaluator.run(model))
                except Exception as e:
                    logger.warning(f"DTW epoch eval failed: {e}")
            else:
                logger.info(
                    f"Skipping DTW eval this epoch (dtw_epoch_eval.eval_every={self.dtw_eval_every}); "
                    "only train/val loss logged."
                )

        self.writer.log(metrics)
        r10 = metrics.get("val_recall@10")
        n10 = metrics.get("val_ndcg@10")
        logger.info(
            f"Epoch {epoch_idx} | train_loss={metrics.get('train_loss')} "
            f"val_loss={metrics.get('val_loss')} "
            f"val_recall@10={r10} val_ndcg@10={n10}"
        )

        if "val_recall@10" in metrics and metrics.get("val_recall@10") is not None:
            return float(metrics["val_recall@10"])
        if val_loss is not None:
            return -float(val_loss)
        return 0.0


class ContrastiveTrainer:
    """Fine-tune a SentenceTransformer with contrastive learning."""

    def __init__(self, conf: Dict):
        self.conf = conf
        ft = conf["finetune"]

        self.base_model = ft["base_model"]
        self.batch_size = ft["batch_size"]
        self.max_seq_length = ft["max_seq_length"]
        self.learning_rate = ft["learning_rate"]
        self.epochs = ft["epochs"]
        self.warmup_ratio = ft["warmup_ratio"]
        self.output_dir = ft["output_dir"]

        self.train_path = conf["data"]["train_jsonl"]
        self.val_path = conf["data"]["val_jsonl"]

    def _load_jsonl(self, path: str) -> List[InputExample]:
        """Load JSONL contrastive pairs as InputExamples for MNRL.

        Each line: {"query": str, "pos": [str], ...}
        The dtw_score field is ignored if present; MNRL uses in-batch negatives.
        """
        examples: List[InputExample] = []
        dropped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    q = data["query"]
                    p = data["pos"][0] if isinstance(data["pos"], list) else data["pos"]
                    if len(q) >= 5 and len(p) >= 5:
                        examples.append(InputExample(texts=[q, p]))
                    else:
                        dropped += 1
                except Exception:
                    dropped += 1
        if dropped:
            logger.warning(f"{path}: dropped {dropped} malformed/short rows")
        logger.info(f"{path}: loaded {len(examples)} examples")
        return examples

    def train(self):
        """Execute the full training pipeline."""
        log_dir = build_log_dir(self.conf)
        writer = MetricsWriter(log_dir)
        _status = os.path.join(log_dir, "run_status.txt")
        logger.info(
            f"Step05 artifacts under: {log_dir}  |  live progress: tail -f {_status!r}"
        )

        # console="off" keeps wandb from wrapping sys.stderr — without it, tqdm often looks "frozen".
        try:
            wandb.init(
                project="nasdaq100-text-finetuning",
                name=f"{self.base_model.split('/')[-1]}-{self.conf['experiment']['NewsAgg']}-finetune",
                config={
                    "branch": self.conf["experiment"]["NewsAgg"],
                    "model": self.conf["experiment"]["EmbModel"],
                    "base_model": self.base_model,
                    "batch_size": self.batch_size,
                    "max_seq_length": self.max_seq_length,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                },
                dir=log_dir,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
        except Exception as e:
            logger.warning(
                f"wandb.init with console=off failed ({e}); tqdm may be invisible. "
                "Set env WANDB_CONSOLE=off before running. Retrying default init."
            )
            wandb.init(
                project="nasdaq100-text-finetuning",
                name=f"{self.base_model.split('/')[-1]}-{self.conf['experiment']['NewsAgg']}-finetune",
                config={
                    "branch": self.conf["experiment"]["NewsAgg"],
                    "model": self.conf["experiment"]["EmbModel"],
                    "base_model": self.base_model,
                    "batch_size": self.batch_size,
                    "max_seq_length": self.max_seq_length,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                },
                dir=log_dir,
                reinit=True,
            )

        logger.info(f"Loading training data from {self.train_path}...")
        train_examples = self._load_jsonl(self.train_path)
        logger.info(f"  Loaded {len(train_examples)} training examples")
        if not train_examples:
            logger.error("No training examples; aborting.")
            wandb.finish()
            return

        val_examples: List[InputExample] = []
        if os.path.exists(self.val_path):
            val_examples = self._load_jsonl(self.val_path)
            logger.info(f"  Loaded {len(val_examples)} validation examples")

        # Guard against NoDuplicatesDataLoader infinite loop: it rejects any
        # example where ANY of its texts already appeared in the current batch.
        # With high duplication, a batch can never be filled. Cap at 90% of the
        # tightest column to account for cross-column overlap.
        _per_col: Dict[int, set] = {}
        for ex in train_examples:
            for i, t in enumerate(ex.texts):
                _per_col.setdefault(i, set()).add(
                    t.strip().lower() if isinstance(t, str) else str(t).strip().lower()
                )
        _min_unique = min(len(v) for v in _per_col.values()) if _per_col else 0
        _safe_bs = max(2, int(_min_unique * 0.9))
        _eff_bs = min(self.batch_size, _safe_bs)
        if _eff_bs < self.batch_size:
            logger.warning(
                f"NoDuplicatesDataLoader: batch_size {self.batch_size} > safe max "
                f"{_safe_bs} (min unique per column={_min_unique}). "
                f"Using batch_size={_eff_bs} to avoid infinite loop."
            )
        train_loader = datasets.NoDuplicatesDataLoader(
            train_examples, batch_size=_eff_bs
        )

        logger.info(f"Loading base model: {self.base_model}")
        word_model = models.Transformer(
            self.base_model,
            max_seq_length=self.max_seq_length,
            model_kwargs={"use_safetensors": True},
        )
        word_model.auto_model.gradient_checkpointing_enable()

        p_mode = "mean" if "contriever" in self.base_model.lower() else "cls"
        pooling = models.Pooling(
            word_model.get_embedding_dimension(), pooling_mode=p_mode
        )
        model = SentenceTransformer(modules=[word_model, pooling])

        mnrl_scale = float(self.conf.get("finetune", {}).get("mnrl_scale", 20.0))
        train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=mnrl_scale)
        logger.info(f"Loss function: MultipleNegativesRankingLoss (scale={mnrl_scale}, temperature={1/mnrl_scale:.4f})")

        # Forward hook on the loss module captures per-batch loss values so we
        # can report a train_loss mean at epoch end.
        tracker = _LossTracker()

        def _hook(_module, _inputs, output):
            if not _module.training:
                return
            try:
                tracker.sum += float(output.detach().item())
                tracker.count += 1
            except Exception:
                pass

        train_loss.register_forward_hook(_hook)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            epoch_evaluator = build_dtw_epoch_evaluator(self.conf, device=dev)
        except Exception as e:
            logger.warning(f"DTW epoch evaluator disabled (init failed): {e}")
            epoch_evaluator = None
        if epoch_evaluator is not None:
            logger.info(
                f"Step07-style epoch eval: split={getattr(epoch_evaluator, 'query_split', '?')} "
                f"{len(getattr(epoch_evaluator, 'eval_gt', {}))} queries, "
                f"corpus rows={len(getattr(epoch_evaluator, 'meta', []))} "
                f"tickers={len(getattr(epoch_evaluator, 'eval_tickers', []))}"
            )

        wrapped_evaluator = _EpochEvalWrapper(
            conf=self.conf,
            val_examples=val_examples,
            train_loss_module=train_loss,
            loss_tracker=tracker,
            writer=writer,
            epoch_evaluator=epoch_evaluator,
            batch_size=self.batch_size,
        )

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        logger.info(
            f"Starting training: {self.epochs} epochs, {total_steps} steps, {warmup_steps} warmup"
        )
        if epoch_evaluator is not None:
            _enc_bs = max(
                1, int(
                    (self.conf.get("finetune", {}).get("dtw_epoch_eval") or {}).get(
                        "encode_batch_size", 64
                    )
                )
            )
            _nrow = len(getattr(epoch_evaluator, "meta", []) or [])
            _n_enc_b = max(1, (_nrow + _enc_bs - 1) // _enc_bs) if _nrow else 0
            logger.info(
                f"Each epoch end: DTW eval shows tqdm on stderr — `DTW corpus encode` (~{_n_enc_b} batches, "
                f"encode_batch_size={_enc_bs}, {_nrow} texts) then `DTW val queries`; long CPU/GPU time is normal."
            )
        else:
            logger.info("DTW epoch eval is off; only train/val loss each epoch.")
        os.makedirs(self.output_dir, exist_ok=True)

        # That line is easy to misread: DTW has NOT started yet. Next is model.fit → first N steps = TRAINING.
        logger.info(
            ">>> Entering model.fit: the FIRST long phase is TRAINING (SentenceTransformers tqdm), "
            "NOT DTW encode. DTW re-encode runs only at each EPOCH END (after all train steps that epoch). "
            "If the screen stays blank, check GPU with nvidia-smi; with wandb, ensure console=off or WANDB_CONSOLE=off."
        )
        print(
            "\n[Capstone] model.fit() starting — first: training steps; then: end-of-epoch DTW eval.\n",
            file=sys.stderr,
            flush=True,
        )

        # dataloader_num_workers=0: worker subprocesses can swallow/delay Ctrl+C; main process loads data.
        _fit = dict(
            train_objectives=[(train_loader, train_loss)],
            evaluator=wrapped_evaluator,
            epochs=self.epochs,
            evaluation_steps=0,
            warmup_steps=warmup_steps,
            output_path=self.output_dir,
            save_best_model=True,
            optimizer_params={"lr": self.learning_rate},
            use_amp=True,
            show_progress_bar=True,
        )

        # Always-readable progress: run_status.txt (tail -f) + stderr lines + heartbeats. Survives broken tqdm/wandb.
        status_path = os.path.join(log_dir, "run_status.txt")
        shared: Dict = {"global_step": 0, "fit_fallback_chain": ""}
        training_header(status_path, total_steps, self.epochs)
        finet = self.conf.get("finetune") or {}
        hb_sec = int(finet.get("heartbeat_sec", 30))
        stop_heartbeat = start_training_heartbeat(
            status_path, hb_sec, shared, total_steps, self.epochs, label="model.fit"
        )
        cbs = make_trainer_step_callbacks(status_path, shared)
        try:
            kwargs: Dict = {
                **_fit,
                "dataloader_num_workers": 0,
                "logging_steps": int(finet.get("logging_steps", 5)),
            }
            if cbs:
                kwargs["callbacks"] = cbs
            try:
                model.fit(**kwargs)
            except TypeError as e1:
                shared["fit_fallback_chain"] = (
                    shared.get("fit_fallback_chain", "") + f" [1] {e1!r}"
                ).strip()
                write_run_status(
                    status_path,
                    f"model.fit TypeError (full kwargs) — will strip unsupported keys and retry. "
                    f"detail: {e1!r}",
                    also_print=True,
                )
                logger.warning(f"model.fit TypeError (full kwargs): {e1!r}")
                err = str(e1).lower()
                if "callback" in err:
                    kwargs.pop("callbacks", None)
                    shared["no_step_sync"] = True
                    write_run_status(
                        status_path,
                        "Stripped: callbacks (HF TrainerCallback unavailable for this ST fit).",
                        also_print=True,
                    )
                    logger.warning(
                        "sentence_transformers.fit() rejected callbacks; "
                        "run_status cannot show optimizer step until ST supports it."
                    )
                if "dataloader" in err:
                    kwargs.pop("dataloader_num_workers", None)
                    write_run_status(
                        status_path, "Stripped: dataloader_num_workers", also_print=True
                    )
                if "logging" in err:
                    kwargs.pop("logging_steps", None)
                    write_run_status(status_path, "Stripped: logging_steps", also_print=True)
                try:
                    model.fit(**kwargs)
                except TypeError as e2:
                    shared["fit_fallback_chain"] = (
                        shared.get("fit_fallback_chain", "") + f" [2] {e2!r}"
                    ).strip()
                    write_run_status(
                        status_path,
                        f"model.fit TypeError (after strip) — retry dataloader_num_workers=0 only. "
                        f"detail: {e2!r}",
                        also_print=True,
                    )
                    logger.warning(f"model.fit TypeError (partial kwargs): {e2!r}")
                    k2 = {**_fit, "dataloader_num_workers": 0}
                    try:
                        model.fit(**k2)
                    except TypeError as e3:
                        shared["fit_fallback_chain"] = (
                            shared.get("fit_fallback_chain", "") + f" [3] {e3!r}"
                        ).strip()
                        shared["no_step_sync"] = True
                        write_run_status(
                            status_path,
                            f"model.fit TypeError (k2) — last resort: SentenceTransformer.fit base "
                            f"kwargs only. detail: {e3!r}",
                            also_print=True,
                        )
                        logger.warning(f"model.fit TypeError (minimal path): {e3!r}")
                        write_run_status(
                            status_path,
                            "model.fit minimal kwargs only; no step sync in run_status.",
                            also_print=True,
                        )
                        model.fit(**_fit)
        finally:
            stop_heartbeat()
            write_run_status(
                status_path, "==== model.fit finished (or interrupted) ====", also_print=True
            )

        writer.plot()
        logger.info(f"Training complete. Best model saved to: {self.output_dir}")
        logger.info(f"Metrics written to {log_dir}")
        wandb.finish()
