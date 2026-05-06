# The overview of structure

```text
Capstone_Recode/
├── README.md            # Project readme, env dependencies, run commands
├── config.yaml          # Core config (hyperparameters, time windows, API settings, etc.)
├── utility.py           # Data loading & preprocessing (Tushare/AkShare clean & align)
├── models/              # Model definitions
│   ├── __init__.py
│   ├── Retriever.py     # Multimodal retriever (news text + price series similarity)
│   └── StockCBR.py      # Core model (integrates retrieved cases for generation & explainable prediction)
├── main.py              # Single entry point (index build, fine-tuning, backtest/eval)
├── datasets/            # Local cached data (cleaned prices, news feature stores)
├── logs/                # Run logs (retrieval accuracy, backtest returns, etc.)
└── checkpoints/         # Saved weights, Faiss indices, or LLM prompt templates
```
