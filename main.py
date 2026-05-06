"""
main.py — Multi-Experiment Entry Point for Capstone_Context
============================================================
Usage:
    python main.py --step 02                          # Run Stage 02
    python main.py --step 03 --NewsAgg agg            # Run Stage 03 on "agg" branch
    python main.py --step 03 --NewsAgg sum            # Run Stage 03 on "sum" branch
    python main.py --step 03 --NewsAgg pool           # Run Stage 03 on "pool" branch
    python main.py --step 05 --NewsAgg pool --EmbModel bge --notify   # Sound + notify when done
    python main.py --step 06 --NewsAgg agg --EmbModel bge --RetModel base   # Pretrained baseline embeddings (no Step 05)
    CAPSTONE_NOTIFY=1 python main.py --step 07 --NewsAgg pool --EmbModel bge
    # POOL text-space 5D variant: use AGG/SUM 5D text encoded as one vector per window
    python main.py --step 03 --NewsAgg pool --RetModel text --EmbModel bge   # produce text-based 5D pool parquet
    python main.py --step 04 --NewsAgg pool --EmbModel bge --variant text5d  # Step 04 reads the text-based 5D pool
    python main.py --step 07 --NewsAgg pool --EmbModel bge --variant text5d  # Step 07 evaluates against it
"""

import argparse
import os
import yaml
from utility import _resolve_vars, get_logger, notify_run_finished

logger = get_logger("main")


def get_args():
    parser = argparse.ArgumentParser(
        description="Capstone_Recode: Multi-Experiment Nasdaq-100 Pipeline"
    )
    parser.add_argument(
        "--step", type=str, required=True,
        choices=["02", "03", "04", "05", "06", "07"],
        help="Pipeline step to run (02-07)."
    )
    parser.add_argument(
        "--RetModel", type=str, default=None,
        help="Optional retrieval model / RetModel-step within the chosen step."
    )
    parser.add_argument(
        "--NewsAgg", type=str, default=None,
        choices=["agg", "sum", "pool"],
        help="News aggregation branch (overrides config.experiment.NewsAgg)."
    )
    parser.add_argument(
        "--EmbModel", type=str, default=None,
        choices=["bge", "contriever"],
        help="Embedding model choice (overrides config.experiment.EmbModel)."
    )
    parser.add_argument(
        "--config", type=str, default="./config.yaml",
        help="Path to config.yaml."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paths without executing."
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="After successful run: play sound + desktop notification (if available). "
        "Or set env CAPSTONE_NOTIFY=1.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["text5d"],
        help=(
            "Optional runtime variant. "
            "'text5d' (pool branch only): downstream steps read the text-based 5D pool "
            "output (nasdaq100_5d_pool_text_<source>_<model>) produced by "
            "`--step 03 --NewsAgg pool --RetModel text`, instead of the daily-pool 5D output."
        ),
    )
    return parser.parse_args()


def load_config_with_override(
    path: str,
    branch_override: str = None,
    model_override: str = None,
    variant: str = None,
):
    """Load config and optionally override active parameters before resolution.

    Variants (post-resolution data-path rewrite):
      - "text5d" + pool branch: point `data.news_5d_summarized_dir` to the text-based
        5D pool output produced by step03 `_run_pool_text5d`, so Step 04/Step 07 pick
        up those vectors without needing signature changes.
    """
    with open(path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    if branch_override:
        conf["experiment"]["NewsAgg"] = branch_override
        logger.info(f"Overriding active branch: {branch_override}")

    if model_override:
        conf["experiment"]["EmbModel"] = model_override
        logger.info(f"Overriding active model: {model_override}")

    conf = _resolve_vars(conf, conf)

    conf.setdefault("runtime", {})["variant"] = variant

    if variant == "text5d":
        branch = conf["experiment"]["NewsAgg"]
        if branch != "pool":
            logger.warning(
                f"--variant text5d only affects the 'pool' branch; current branch is '{branch}'. Ignoring."
            )
        else:
            pool_conf = conf.get("pool", {}) or {}
            text_source = pool_conf.get("text5d_source", "sum")
            if text_source not in ("agg", "sum"):
                raise ValueError(
                    f"pool.text5d_source must be 'agg' or 'sum', got: {text_source!r}"
                )
            EmbModel = conf["experiment"]["EmbModel"]
            dataset_dir = conf["data"]["dataset_dir"]
            new_dir = os.path.join(
                dataset_dir,
                f"nasdaq100_5d_pool_text_{text_source}_{EmbModel}",
            )
            old_dir = conf["data"].get("news_5d_summarized_dir")
            conf["data"]["news_5d_summarized_dir"] = new_dir
            logger.info(
                f"[variant=text5d] overriding data.news_5d_summarized_dir: {old_dir} -> {new_dir}"
            )

    return conf


def main():
    args = get_args()
    conf = load_config_with_override(args.config, args.NewsAgg, args.EmbModel, args.variant)

    NewsAgg = conf["experiment"]["NewsAgg"]
    EmbModel = conf["experiment"]["EmbModel"]

    # Set CUDA device
    cuda_dev = conf.get("device", {}).get("cuda_visible_devices", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)

    logger.info(
        f"RUN: step={args.step} | NewsAgg={NewsAgg} | EmbModel={EmbModel} | "
        f"RetModel={args.RetModel} | variant={args.variant}"
    )

    if args.dry_run:
        _validate(conf, args.step)
        return

    # Dynamic dispatch
    if args.step == "02":
        from pipeline.step02_preprocess import run
    elif args.step == "03":
        from pipeline.step03_dtw_summarize import run
    elif args.step == "04":
        from pipeline.step04_generate_pairs import run
    elif args.step == "05":
        from pipeline.step05_finetune import run
    elif args.step == "06":
        from pipeline.step06_embedding import run
    elif args.step == "07":
        from pipeline.step07_evaluate import run
    else:
        raise ValueError(f"Unknown step: {args.step}")

    run(conf, RetModel=args.RetModel)
    logger.info(f"Step {args.step} ({NewsAgg}) completed.")
    want_notify = args.notify or os.environ.get("CAPSTONE_NOTIFY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if want_notify:
        variant_tag = f", variant={args.variant}" if args.variant else ""
        notify_run_finished(
            "Capstone_Recode",
            f"Step {args.step} ({NewsAgg}, {EmbModel}{variant_tag}) finished OK.",
        )


def _validate(conf: dict, step: str):
    """Path validation for the active branch."""
    logger.info("=== Dry Run: Validating paths ===")
    data = conf.get("data", {})
    branch = conf["experiment"]["NewsAgg"]
    
    checks = {
        "02": ["anomalies_per_stock_dir", "news_full_dir"],
        "03": ["prices_dir", "news_full_dir"],
        "04": ["dtw_results_dir"],
        "05": ["train_jsonl", "val_jsonl"],
        "06": ["sliding_windows_dir"],
        "07": ["dtw_results_dir", "embeddings_dir"],
    }
    
    required = list(checks.get(step, []))
    if step == "07":
        if branch == "pool":
            required = ["dtw_results_dir", "news_5d_summarized_dir"]
        else:
            ev = conf.get("evaluate", {})
            if ev.get("dense_embedding_source", "finetuned") == "pretrained":
                required = ["dtw_results_dir", "embeddings_pretrained_dir"]
            else:
                required = ["dtw_results_dir", "embeddings_dir"]

    all_ok = True
    for key in required:
        path = data.get(key, "")
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        logger.info(f"  {status} [{branch}] {key}: {path}")
        if not exists:
            all_ok = False
            
    if all_ok:
        logger.info(f"Branch [{branch}] readiness: OK")
    else:
        logger.warning(f"Branch [{branch}] readiness: FAILED. Check paths/folders.")


if __name__ == "__main__":
    main()