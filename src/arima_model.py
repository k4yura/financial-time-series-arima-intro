"""
arima_model.py

ARIMA scaffold using statsmodels.

Students TODOs:
- implement select_order_grid(...) to search small (p,d,q) grid using AIC/BIC
- add walk-forward evaluation if desired
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA

def select_order_grid(series: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int,int,int]:
    """
    Very small grid search for (p,d,q) based on AIC.
    TODO: Students implement search; current placeholder returns (1,0,1) or (1,1,1) based on ADF pvalue.

    NOTE: keep grid small for speed in class.
    """
    # Placeholder simple heuristic; students should replace with real search
    try:
        from statsmodels.tsa.stattools import adfuller
        pval = adfuller(series.dropna(), autolag="AIC")[1]
        d = 0 if pval < 0.05 else 1
    except Exception:
        d = 1
    return (1, d, 1)

def fit_arima(series: pd.Series, order: Tuple[int,int,int], enforce_stationarity: bool = False, enforce_invertibility: bool = False):
    """
    Fit ARIMA and return fitted result object.
    Students: add robust try/except and parameter tuning.
    """
    model = ARIMA(series, order=order, enforce_stationarity=enforce_stationarity,
                  enforce_invertibility=enforce_invertibility)
    res = model.fit()
    return res

def forecast_from_model(fitted_model, steps: int = 5) -> pd.Series:
    """Return point forecast for `steps` periods ahead."""
    return fitted_model.forecast(steps=steps)
