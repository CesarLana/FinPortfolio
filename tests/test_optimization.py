"""
Unit tests for optimization module.
"""

import pytest
import pandas as pd
import numpy as np

from src.optimization import (
    optimize_gmv,
    optimize_max_sharpe,
    optimize_target_return,
    calculate_efficient_frontier
)


@pytest.fixture
def optimization_data():
    exp_returns = pd.Series([0.15, 0.10, 0.20], index=["A", "B", "C"])
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01, 0.02],
         [0.01, 0.025, 0.005],
         [0.02, 0.005, 0.06]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"]
    )
    return exp_returns, cov_matrix


def test_optimize_gmv(optimization_data):
    exp_returns, cov_matrix = optimization_data
    res = optimize_gmv(exp_returns, cov_matrix, min_weight=0.0, max_weight=1.0)

    weights = res["weights"]
    assert np.isclose(weights.sum(), 1.0, atol=1e-4)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()
    assert res["volatility"] > 0.0


def test_optimize_max_sharpe(optimization_data):
    exp_returns, cov_matrix = optimization_data
    rf = 0.05
    res = optimize_max_sharpe(exp_returns, cov_matrix, risk_free_rate=rf)

    weights = res["weights"]
    assert np.isclose(weights.sum(), 1.0, atol=1e-4)
    assert (weights >= -1e-5).all()
    assert res["sharpe"] > 0.0


def test_optimize_max_sharpe_with_bounds(optimization_data):
    exp_returns, cov_matrix = optimization_data
    rf = 0.05
    min_w, max_w = 0.10, 0.50
    res = optimize_max_sharpe(exp_returns, cov_matrix, risk_free_rate=rf, min_weight=min_w, max_weight=max_w)

    weights = res["weights"]
    assert np.isclose(weights.sum(), 1.0, atol=1e-3)
    assert (weights >= min_w - 1e-4).all()
    assert (weights <= max_w + 1e-4).all()


def test_calculate_efficient_frontier(optimization_data):
    exp_returns, cov_matrix = optimization_data
    frontier = calculate_efficient_frontier(exp_returns, cov_matrix, num_points=10)

    assert len(frontier) > 0
    assert "return" in frontier.columns
    assert "volatility" in frontier.columns
    assert "sharpe" in frontier.columns
