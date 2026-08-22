"""
Unit tests for returns module.
"""

import pytest
import pandas as pd
import numpy as np
from src.returns import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_annualized_returns,
    calculate_portfolio_daily_returns
)


@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = {
        "ASSET1": [100.0, 102.0, 101.0, 105.0, 110.0],
        "ASSET2": [50.0, 49.0, 51.0, 52.0, 55.0]
    }
    return pd.DataFrame(data, index=dates)


def test_calculate_daily_returns_arithmetic(sample_prices):
    ret = calculate_daily_returns(sample_prices, method='arithmetic')
    assert len(ret) == 4
    assert np.isclose(ret["ASSET1"].iloc[0], 0.02)  # (102-100)/100
    assert np.isclose(ret["ASSET2"].iloc[0], -0.02)  # (49-50)/50


def test_calculate_daily_returns_log(sample_prices):
    ret = calculate_daily_returns(sample_prices, method='log')
    assert len(ret) == 4
    assert np.isclose(ret["ASSET1"].iloc[0], np.log(102/100))


def test_calculate_cumulative_returns(sample_prices):
    daily_ret = calculate_daily_returns(sample_prices, method='arithmetic')
    cum_ret = calculate_cumulative_returns(daily_ret, method='arithmetic')
    # Final cumulative return should equal (Final Price - Initial Price) / Initial Price
    assert np.isclose(cum_ret["ASSET1"].iloc[-1], (110.0 - 100.0) / 100.0)
    assert np.isclose(cum_ret["ASSET2"].iloc[-1], (55.0 - 50.0) / 50.0)


def test_calculate_annualized_returns(sample_prices):
    daily_ret = calculate_daily_returns(sample_prices, method='arithmetic')
    ann_ret = calculate_annualized_returns(daily_ret, trading_days=252, method='arithmetic')
    assert len(ann_ret) == 2
    assert ann_ret["ASSET1"] > 0


def test_calculate_portfolio_daily_returns(sample_prices):
    daily_ret = calculate_daily_returns(sample_prices, method='arithmetic')
    weights = [0.5, 0.5]
    port_ret = calculate_portfolio_daily_returns(daily_ret, weights)
    expected_first = 0.5 * 0.02 + 0.5 * (-0.02)
    assert np.isclose(port_ret.iloc[0], expected_first)
