"""
Unit tests for data module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data import normalize_prices, fetch_historical_data


def test_normalize_prices():
    dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [10.0, 15.0, 20.0], "B": [100.0, 110.0, 120.0]}, index=dates)
    norm = normalize_prices(df, base_value=100.0)

    assert norm["A"].iloc[0] == 100.0
    assert norm["A"].iloc[1] == 150.0
    assert norm["A"].iloc[2] == 200.0
    assert norm["B"].iloc[0] == 100.0


def test_fetch_historical_data_invalid_input():
    with pytest.raises(ValueError):
        fetch_historical_data([], "2023-01-01", "2023-06-01")
