# financial-time-series-arima-intro

Part 1: Data ingestion → EDA → ARIMA scaffold (statsmodels, yfinance)

This repository is a simplified, code-focused workshop starter.  
Students are expected to implement the TODOs marked in the code.

## Quickstart

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the demo pipeline (auto-pulls SPY last 2 years by default):
```bash
python src/main.py
```

3. Inspect and implement TODOs in `src/eda.py` and `src/arima_model.py`.

## Structure
```
financial-time-series-arima-intro/
├── README.md
├── requirements.txt
└── src/
    ├── data_loader.py
    ├── eda.py
    ├── arima_model.py
    └── main.py
```

