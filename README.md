# Attention-Based-Multi-Asset-Forecasting-with-News-Driven-Criteria

 

## ✨ Key Features

-   **Multi-modal Data Fusion**: Aligns 5-minute price bars with timestamped financial news headlines
-   **FinBERT Integration**: Domain-specific language model fine-tuned for financial sentiment and entity extraction
-   **Temporal Fusion Transformer (TFT)**: State-of-the-art interpretable architecture for multi-horizon forecasting
-   **Attention-Based Interpretability**: Multi-level attention extraction (variable selection, temporal, token-level)
-   **Criteria Dictionary Builder**: Automatically surfaces decision-driving phrases with directional impact statistics
-   **Reproducible Experiments**: Chronological train/val/test splits with comprehensive ablation studies

---

## 📦 Installation

### Prerequisites
- Python ≥ 3.9
- CUDA-capable GPU (recommended for training)
- ~50GB disk space for data and models

### Setup

```bash
# Clone repository
git clone https://github.com/yamenetoo/Attention-Based-Multi-Asset-Forecasting-with-News-Driven-Criteria.git
cd commodity-forecasting-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download FinBERT (auto-downloaded on first run)
# Or manually:
python -m src.models.download_models --model finbert
```
 

## 🗂️ Project Structure

```
commodity-forecasting-attention/
├── README.md                 
├── LICENSE                   
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment 
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config/               # Configuration files
│   │   ├── default.yaml
│   │   ├── model.yaml
│   │   └── data.yaml
│   │
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py         # Price + news data loader
│   │   ├── aligner.py        # Timestamp alignment logic
│   │   ├── preprocess.py     # Cleaning, normalization, feature engineering
│   │   └── dataset.py        # PyTorch Dataset classes
│   │
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   ├── finbert_wrapper.py    # FinBERT encoding interface
│   │   ├── tft.py                # Temporal Fusion Transformer implementation
│   │   ├── attention_utils.py    # Attention extraction utilities
│   │   └── criteria_extractor.py # Phrase impact scoring & dictionary builder
│   │
│   ├── training/             # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop with validation
│   │   ├── loss.py           # Quantile loss, directional loss
│   │   └── callbacks.py      # Early stopping, checkpointing
│   │
│   ├── evaluation/           # Metrics & analysis
│   │   ├── __init__.py
│   │   ├── metrics.py        # RMSE, MAE, directional accuracy
│   │   ├── backtest.py       # Trading simulation utilities
│   │   └── visualization.py  # Plotting functions for figures
│   │
│   └── utils/                # Helper functions
│       ├── __init__.py
│       ├── logger.py
│       ├── seed.py           # Reproducibility utilities
│       └── io.py             # File I/O helpers
│
├── paper/                    # LaTeX source for manuscript
│   ├── main.tex              # Main document
│   ├── references.bib        # Bibliography
│   ├── figures/              # TikZ figures and assets
│   └── compile.sh            # Compilation script
│
├── data/                     # Data directory (git-ignored)
│   ├── raw/                  # Original data (downloaded separately)
│   │   ├── news/             # JSONL news headlines with timestamps
│   │   ├── prices/           # CSV price data (XAU/USD, WTI)
│   │   └── entities/         # NER tag dictionaries
│   │
│   ├── processed/            # Preprocessed datasets
│   │   ├── aligned/          # Timestamp-aligned price+news pairs
│   │   ├── embeddings/       # Cached FinBERT embeddings
│   │   └── splits/           # Train/val/test indices
│   │
│   └── external/             # Third-party data sources
│
├── models/                   # Saved model checkpoints (git-ignored)
│   ├── finbert/
│   ├── tft_news_gold/
│   ├── tft_news_oil/
│   └── criteria_dictionaries/
│
├── results/                  # Experiment outputs
│   ├── metrics/              # CSV logs of evaluation metrics
│   ├── figures/              # Generated plots (PNG/PDF)
│   ├── criteria/             # Extracted phrase dictionaries
│   └── backtests/            # Trading simulation results
│
├── scripts/                  # Command-line entry points
│   ├── preprocess.py         # Run data alignment & preprocessing
│   ├── train.py              # Train TFT model
│   ├── evaluate.py           # Evaluate on test set
│   ├── extract_criteria.py   # Build criteria dictionary from attention
│   ├── backtest.py           # Run trading simulation
│   └── generate_figures.py   # Reproduce paper figures
│
└── tests/                    # Unit tests
    ├── test_data.py
    ├── test_models.py
    └── test_attention.py
```

---

## 🚀 Quick Start

### 1. Data Preparation
```bash
# Download sample data (or provide your own)
python scripts/download_sample_data.py

# Preprocess and align news with price data
python scripts/preprocess.py \
  --news-path data/raw/news/ \
  --price-path data/raw/prices/ \
  --output-path data/processed/aligned/ \
  --assets gold oil \
  --frequency 5min \
  --horizon 30min
```

### 2. Training
```bash
# Train TFT model with news embeddings (Gold)
python scripts/train.py \
  --config src/config/default.yaml \
  --asset gold \
  --use-news \
  --output-dir models/tft_news_gold/ \
  --gpus 0

# Train for Oil (or both jointly with --multi-asset)
python scripts/train.py --asset oil --use-news
```

### 3. Evaluation
```bash
# Evaluate on test set
python scripts/evaluate.py \
  --model-path models/tft_news_gold/best.pt \
  --test-split data/processed/splits/gold_test.pkl \
  --output results/metrics/gold_test.json

# Generate figures for paper
python scripts/generate_figures.py --config paper/figures_config.yaml
```

### 4. Criteria Extraction
```bash
# Extract price-moving phrases from attention weights
python scripts/extract_criteria.py \
  --model-path models/tft_news_gold/best.pt \
  --data-path data/processed/aligned/gold/ \
  --min-freq 100 \
  --max-phrase-length 5 \
  --output results/criteria/gold_dictionary.json
```

### 5. Backtesting (Optional)
```bash
# Simulate criteria-based trading strategy
python scripts/backtest.py \
  --criteria results/criteria/gold_dictionary.json \
  --price-data data/processed/prices/gold_test.csv \
  --transaction-cost 0.0001 \
  --output results/backtests/gold_criteria_strategy.json
```

---

## ⚙️ Configuration

Key parameters in `src/config/default.yaml`:

```yaml
data:
  assets: ["gold", "oil"]
  price_frequency: "5min"
  forecast_horizon: "30min"  # prediction target
  lookback_window: 24        # number of past bars (2 hours)
  news_lookback: "5min"      # news window before each bar

model:
  finbert:
    model_name: "ProsusAI/finbert"
    max_length: 128
    freeze_embeddings: false
  
  tft:
    hidden_size: 128
    attention_head_count: 4
    lstm_layers: 2
    dropout: 0.1
    quantiles: [0.05, 0.5, 0.95]  # for probabilistic forecasts

training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 100
  early_stopping_patience: 10
  loss_type: "quantile"  # or "mse", "directional"

interpretability:
  extraction:
    min_phrase_frequency: 100
    max_ngram_length: 5
    attention_threshold: 0.01
```

---

## 📊 Reproducing Paper Results

All tables and figures from the manuscript can be regenerated:

```bash
# Full reproduction pipeline
bash scripts/reproduce_all.sh

# Or step-by-step:
# 1. Preprocess data
python scripts/preprocess.py --config src/config/paper_config.yaml

# 2. Train models (requires ~8GB GPU memory)
python scripts/train.py --asset gold --seed 42
python scripts/train.py --asset oil --seed 42

# 3. Evaluate and generate tables
python scripts/evaluate.py --all
python scripts/generate_tables.py --output paper/tables/

# 4. Generate figures (TikZ-compatible data exports)
python scripts/generate_figures.py --format pdf --output paper/figures/
```

> ⏱️ **Runtime Estimates** (on NVIDIA RTX 3090):
> - Preprocessing: ~2 hours
> - Training per asset: ~4-6 hours
> - Criteria extraction: ~30 minutes
> - Full reproduction: ~12-15 hours

---

## 🔍 Understanding the Attention-Based Criteria Extraction

The interpretability pipeline computes a **Word Impact Score**:

```
Impact(w_i) = α_FinBERT(w_i) × β_VSN(N_k) × sgn(ŷ_t)
```

Where:
- `α_FinBERT(w_i)`: Token-level attention from FinBERT's last layer
- `β_VSN(N_k)`: Variable selection weight for news item N_k containing word w_i
- `sgn(ŷ_t)`: Predicted return direction (+1/-1)

This score is aggregated across n-grams (1-5 words), filtered by frequency, and used to build a **Criteria Dictionary** containing:
- Phrase text and entity category (via NER)
- Directional consistency (% of times phrase predicted correct direction)
- Average absolute price move (basis points) following phrase occurrence
- Temporal decay profile (immediate vs. delayed impact)

Example output (`results/criteria/gold_dictionary.json`):
```json
{
  "Fed raises interest rates": {
    "direction": "negative",
    "consistency": 0.821,
    "avg_abs_move_bps": 4.2,
    "entity_category": "monetary_policy",
    "occurrences": 342,
    "attention_profile": {"0-5min": 0.45, "5-15min": 0.32, "15-30min": 0.23}
  }
}
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration test (small dataset)
pytest tests/integration/ -v --slow

# Check code style
flake8 src/ scripts/
black --check src/ scripts/
```

---

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{almohamad2026attention,
  title={Attention-Based Multi-Asset Forecasting with News-Driven Criteria Extraction: A Case Study on Gold and Crude Oil},
  author={Al-Mohamad, Mohamad Yamen and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
 

---
 
