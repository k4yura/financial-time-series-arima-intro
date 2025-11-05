"""
eda.py

Exploratory Data Analysis helpers for time series.

Students should implement plotting and stationarity checks.
Functions:
- compute_log_returns(series)
- plot_series(series)   (simple matplotlib plotting)
- adf_test_wrapper(series)  (calls statsmodels adfuller and returns p-value)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def compute_log_returns(series: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.
    TODO: handle zeros / negative values robustly if present.
    """
    return np.log(series).diff().dropna()

def plot_series(series: pd.Series, title: str = "series"):
    """Simple line plot of a series. Students may save or show the plot."""
    plt.figure(figsize=(10,4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

def adf_test_wrapper(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    Run Augmented Dickey-Fuller test and return basic info.
    Returns: {'adf_stat':..., 'pvalue':..., 'crit_vals': {...}}
    """
    res = adfuller(series.dropna(), autolag=autolag)
    return {"adf_stat": res[0], "pvalue": res[1], "crit_vals": res[4]}
