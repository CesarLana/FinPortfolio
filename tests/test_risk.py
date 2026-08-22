"""
Unit tests for risk module.
"""

import pytest
import pandas as pd
import numpy as np

from src.risk import (
    calculate_volatility,
    calculate_variance,
    calculate_covariance_matrix,
    calculate_correlation_matrix,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_beta_and_alpha,
    calculate_diversification_metrics,
    calculate_extreme_returns
)


@pytest.fixture
def sample_daily_returns():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = {
        "ASSET1": np.random.normal(0.001, 0.015, 100),
        "ASSET2": np.random.normal(0.0005, 0.010, 100)
    }
    return pd.DataFrame(data, index=dates)


def test_volatility_and_variance(sample_daily_returns):
    daily_vol, ann_vol = calculate_volatility(sample_daily_returns, trading_days=252)
    ann_var = calculate_variance(sample_daily_returns, trading_days=252)

    assert len(daily_vol) == 2
    assert (ann_vol > daily_vol).all()
    assert np.isclose(ann_var["ASSET1"], ann_vol["ASSET1"] ** 2)


def test_covariance_and_correlation_matrices(sample_daily_returns):
    cov_mat = calculate_covariance_matrix(sample_daily_returns, trading_days=252)
    corr_mat = calculate_correlation_matrix(sample_daily_returns)

    assert cov_mat.shape == (2, 2)
    assert corr_mat.shape == (2, 2)
    assert np.isclose(corr_mat.iloc[0, 0], 1.0)
    assert np.isclose(corr_mat.iloc[1, 1], 1.0)
    assert np.isclose(corr_mat.iloc[0, 1], corr_mat.iloc[1, 0])


def test_portfolio_volatility(sample_daily_returns):
    cov_mat = calculate_covariance_matrix(sample_daily_returns, trading_days=252)
    weights = [0.6, 0.4]
    port_vol = calculate_portfolio_volatility(weights, cov_mat)

    assert port_vol > 0.0
    assert isinstance(port_vol, float)


def test_sharpe_and_sortino_ratio(sample_daily_returns):
    sharpe = calculate_sharpe_ratio(expected_return=0.15, volatility=0.20, risk_free_rate=0.05)
    assert np.isclose(sharpe, 0.50)

    sortino = calculate_sortino_ratio(sample_daily_returns["ASSET1"], expected_return=0.15, risk_free_rate=0.05)
    assert isinstance(sortino, float)


def test_calmar_ratio():
    calmar = calculate_calmar_ratio(cagr=0.20, max_drawdown=-0.10)
    assert np.isclose(calmar, 2.0)


def test_max_drawdown():
    prices = pd.Series([100.0, 120.0, 90.0, 110.0])
    mdd = calculate_max_drawdown(prices)
    assert np.isclose(mdd, -0.25)


def test_value_at_risk_and_cvar(sample_daily_returns):
    var_95 = calculate_value_at_risk(sample_daily_returns["ASSET1"], confidence_level=0.95)
    var_99 = calculate_value_at_risk(sample_daily_returns["ASSET1"], confidence_level=0.99)
    cvar_95 = calculate_cvar(sample_daily_returns["ASSET1"], confidence_level=0.95)

    assert var_99 <= var_95
    assert cvar_95 <= var_95


def test_beta_and_alpha(sample_daily_returns):
    beta, alpha = calculate_beta_and_alpha(
        sample_daily_returns["ASSET1"],
        sample_daily_returns["ASSET2"],
        risk_free_rate=0.05
    )
    assert isinstance(beta, float)
    assert isinstance(alpha, float)


def test_diversification_metrics():
    # 2 assets equally weighted -> HHI = 0.5^2 + 0.5^2 = 0.5, ENC = 2.0
    weights = [0.5, 0.5]
    div = calculate_diversification_metrics(weights)
    assert np.isclose(div['hhi'], 0.5)
    assert np.isclose(div['effective_n'], 2.0)


def test_extreme_returns(sample_daily_returns):
    best, worst = calculate_extreme_returns(sample_daily_returns["ASSET1"])
    assert best > worst
    assert best == sample_daily_returns["ASSET1"].max()
    assert worst == sample_daily_returns["ASSET1"].min()
