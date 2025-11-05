"""
data_loader.py

Simple yfinance-based loader. Auto-download convenience function included.

Functions:
- load_price_data(ticker, start, end): returns DataFrame
- default_pull(ticker="SPY", years=2): convenience wrapper that saves nothing and returns df

Notes:
- Keep this file simple. Students will add cleaning / resampling as exercises.
"""

from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd

def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for ticker between start and end (YYYY-MM-DD strings).

    Returns a DataFrame indexed by date with columns: ['Open','High','Low','Close','Adj Close','Volume']
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    return df

def default_pull(ticker: str = "SPY", years: int = 2) -> pd.DataFrame:
    """Pull last `years` years for ticker."""
    end = datetime.today().date()
    start = end - timedelta(days=365 * years)
    return load_price_data(ticker, start.isoformat(), end.isoformat())

if __name__ == "__main__":
    df = default_pull()
    print(df.tail())
