"""
Unit tests for backtest module.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtest import run_historical_backtest


@pytest.fixture
def backtest_returns():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(123)
    data = {
        "ASSET1": np.random.normal(0.001, 0.015, 100),
        "ASSET2": np.random.normal(0.0008, 0.012, 100)
    }
    df = pd.DataFrame(data, index=dates)
    weights = pd.Series([0.6, 0.4], index=["ASSET1", "ASSET2"])
    bench = pd.Series(np.random.normal(0.0005, 0.010, 100), index=dates, name="BOVA11")
    return df, weights, bench


def test_run_historical_backtest(backtest_returns):
    daily_ret, weights, bench = backtest_returns
    res = run_historical_backtest(
        daily_returns=daily_ret,
        optimized_weights=weights,
        benchmark_returns=bench,
        risk_free_rate=0.1075,
        initial_investment=10000.0
    )

    assert "wealth_curves" in res
    assert "cumulative_returns" in res
    assert "drawdowns" in res
    assert "metrics" in res

    metrics = res["metrics"]
    assert "Carteira Otimizada" in metrics.index
    assert "Carteira 1/N (Igual)" in metrics.index
    assert "Benchmark" in metrics.index
    assert "Retorno Total (%)" in metrics.columns
    assert "Maximum Drawdown (%)" in metrics.columns
