"""
main.py

Orchestrates a minimal pipeline:
1. Pull data with yfinance (default SPY 2 years)
2. Compute log returns
3. Run simple ADF check and pick ARIMA order (scaffold)
4. Fit ARIMA and forecast next N days

Run:
    python src/main.py

Students: edit parameters and fill TODOs in arima_model.py and eda.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import pandas as pd

from src.data_loader import default_pull
from src.eda import compute_log_returns, adf_test_wrapper, plot_series
from src.arima_model import select_order_grid, fit_arima, forecast_from_model

def run_pipeline(ticker: str = "SPY", years: int = 2, forecast_steps: int = 5, plot: bool = False):
    print(f"Pulling {ticker} for last {years} years...")
    df = default_pull(ticker=ticker, years=years)
    if df.empty:
        print("No data returned. Exiting.")
        return
    # Use Adjusted Close for price series
    if "Adj Close" in df.columns:
        price = df["Adj Close"].dropna()
    else:
        price = df["Close"].dropna()

    print("Computing log returns...")
    returns = compute_log_returns(price)

    print("ADF test on returns...")
    adf_res = adf_test_wrapper(returns)
    print(f"ADF p-value: {adf_res['pvalue']:.4f}")

    if plot:
        print("Plotting price (last 200 points)...")
        plot_series(price.tail(200), title=f"{ticker} Price (tail)")

    print("Selecting ARIMA order (scaffold)...")
    order = select_order_grid(returns)
    print(f"Selected order: {order}")

    print("Fitting ARIMA...")
    model = fit_arima(returns, order)
    print(model.summary())

    print(f"Forecasting next {forecast_steps} steps...")
    fc = forecast_from_model(model, steps=forecast_steps)
    print(fc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run_pipeline(ticker=args.ticker, years=args.years, forecast_steps=args.steps, plot=args.plot)
