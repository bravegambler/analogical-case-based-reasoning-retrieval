"""
Step 05 — Multi-Branch Fine-tuning Dispatcher
==============================================
Routes to different trainers based on the active branch:
- AGG / SUM: ContrastiveTrainer (Text-based, MNRL)
- POOL     : VectorBasedTrainer (Vector-based, InfoNCE)
"""

from typing import Dict, Optional
from utility import get_logger

logger = get_logger("step05")


def run(conf: Dict, RetModel: Optional[str] = None):
    branch = conf["experiment"]["NewsAgg"]
    
    logger.info(f"Starting fine-tuning for branch: {branch}")
    
    if branch in ["agg", "sum"]:
        from models.ContrastiveTrainer import ContrastiveTrainer
        trainer = ContrastiveTrainer(conf)
        trainer.train()
    elif branch == "pool":
        from models.VectorBasedTrainer import VectorBasedTrainer
        trainer = VectorBasedTrainer(conf)
        trainer.train()
    else:
        raise ValueError(f"Unknown branch: {branch}")
