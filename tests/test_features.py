"""
Basic tests for feature engineering functions.
"""

import pandas as pd
import numpy as np
from src.feature_engineering import create_lag_features, create_rolling_features

def test_create_lag_features_shape():
    s = pd.Series(range(10))
    df = create_lag_features(s, lags=[1,2,3])
    # original + 3 lag cols -> 4 columns
    assert "y" in df.columns
    assert "lag_1" in df.columns
    assert df.shape[1] == 4

def test_create_rolling_features_contains_columns():
    s = pd.Series(range(20))
    df = create_rolling_features(s, windows=[3,5], funcs=["mean","std"])
    assert any("roll_3_mean" in c for c in df.columns)
    assert any("roll_5_std" in c for c in df.columns)
