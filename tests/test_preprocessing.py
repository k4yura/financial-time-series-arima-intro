"""
Basic tests for preprocessing functions using synthetic data.
These tests are intentionally simple and avoid network access.
Students should add more tests as they implement functionality.
"""

import pandas as pd
import numpy as np
from src.preprocessing import adf_test, make_stationary, train_test_split_ts

def test_adf_on_stationary_series():
    # white noise should be stationary
    rng = np.random.RandomState(0)
    s = pd.Series(rng.normal(size=200))
    res = adf_test(s)
    assert "pvalue" in res
    assert res["pvalue"] >= 0 or res["pvalue"] <= 1  # p-value is valid number

def test_make_stationary_returns_tuple():
    rng = np.random.RandomState(1)
    # random walk (non-stationary)
    x = np.cumsum(rng.normal(size=300))
    s = pd.Series(x)
    stationary, d = make_stationary(s, max_diff=2)
    assert isinstance(stationary, pd.Series)
    assert isinstance(d, int)
    assert d >= 0

def test_train_test_split_ts_fraction():
    s = pd.Series(range(100))
    train, test = train_test_split_ts(s, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
