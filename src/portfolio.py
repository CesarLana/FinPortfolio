"""
Portfolio Simulation Module for FinPortfolio.
Performs Monte Carlo simulation of random asset weight allocations
and evaluates portfolio risk-return metrics.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.returns import (
    calculate_annualized_returns,
    calculate_portfolio_daily_returns
)
from src.risk import (
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_diversification_metrics,
    calculate_extreme_returns
)


def _generate_bounded_weights(
    num_assets: int,
    num_samples: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generates random weight vectors strictly satisfying min_weight <= w_i <= max_weight and sum(w) = 1.
    Uses Dirichlet sampling on the shifted simplex with iterative boundary projection.
    """
    np.random.seed(seed)

    min_w = max(0.0, min_weight)
    max_w = min(1.0, max_weight)

    # Feasibility adjustment if user entered contradictory bounds
    if min_w * num_assets > 1.0:
        min_w = 1.0 / num_assets
    if max_w * num_assets < 1.0:
        max_w = 1.0

    if min_w <= 0.0 and max_w >= 1.0:
        return np.random.dirichlet(np.ones(num_assets), size=num_samples)

    remaining_capacity = 1.0 - num_assets * min_w

    if remaining_capacity <= 1e-6:
        # Uniform equal weights only
        return np.full((num_samples, num_assets), 1.0 / num_assets)

    # Base allocations
    weights_list = []
    # Sample from Dirichlet and adjust by excess capacity
    raw_dirichlet = np.random.dirichlet(np.ones(num_assets), size=num_samples * 2)

    for raw_w in raw_dirichlet:
        w_candidate = min_w + raw_w * remaining_capacity

        # Check if candidate exceeds max_w
        if (w_candidate <= max_w + 1e-5).all():
            weights_list.append(w_candidate / np.sum(w_candidate))
            if len(weights_list) == num_samples:
                break

    # Fallback if rejection yielded fewer samples: bounded iterative clipping
    if len(weights_list) < num_samples:
        needed = num_samples - len(weights_list)
        extra_raw = np.random.dirichlet(np.ones(num_assets), size=needed)
        for w in extra_raw:
            w_adj = min_w + w * remaining_capacity
            w_clipped = np.clip(w_adj, min_w, max_w)
            w_norm = w_clipped / np.sum(w_clipped)
            weights_list.append(w_norm)

    return np.array(weights_list[:num_samples])


def simulate_random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = 0.1075,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulates thousands of random portfolio allocations via Monte Carlo sampling.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    num_portfolios : int, optional
        Number of random portfolios to generate (default 5000).
    risk_free_rate : float, optional
        Risk-free rate (default 0.1075).
    min_weight : float, optional
        Minimum weight per asset.
    max_weight : float, optional
        Maximum weight per asset.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        DataFrame with columns ['return', 'volatility', 'sharpe'], and weights array shape (num_portfolios, num_assets).
    """
    num_assets = len(expected_returns)
    weights_matrix = _generate_bounded_weights(
        num_assets=num_assets,
        num_samples=num_portfolios,
        min_weight=min_weight,
        max_weight=max_weight,
        seed=seed
    )

    returns_arr = np.dot(weights_matrix, expected_returns.values)
    vols_arr = np.array([
        calculate_portfolio_volatility(w, cov_matrix) for w in weights_matrix
    ])

    sharpe_arr = np.where(
        vols_arr > 1e-8,
        (returns_arr - risk_free_rate) / vols_arr,
        0.0
    )

    results_df = pd.DataFrame({
        'return': returns_arr,
        'volatility': vols_arr,
        'sharpe': sharpe_arr
    })

    return results_df, weights_matrix


def calculate_portfolio_metrics_summary(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    risk_free_rate: float = 0.1075,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Computes a comprehensive institutional summary of risk and return metrics for a specific portfolio weight vector.

    Parameters
    ----------
    weights : pd.Series
        Asset weight Series.
    daily_returns : pd.DataFrame
        Daily returns DataFrame.
    risk_free_rate : float, optional
        Risk-free rate.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    Dict[str, float]
        Dictionary containing expected return, volatility, variance, Sharpe ratio,
        Sortino ratio, Calmar ratio, Max Drawdown, VaR 95%, VaR 99%, CVaR 95%,
        best day, worst day, HHI and Effective Number of Assets.
    """
    port_daily_ret = calculate_portfolio_daily_returns(daily_returns, weights.values)
    mean_daily = port_daily_ret.mean()
    ann_return = (1 + mean_daily) ** trading_days - 1
    daily_vol = port_daily_ret.std()
    ann_vol = daily_vol * np.sqrt(trading_days)
    ann_var = ann_vol ** 2

    sharpe = calculate_sharpe_ratio(ann_return, ann_vol, risk_free_rate)
    sortino = calculate_sortino_ratio(port_daily_ret, ann_return, risk_free_rate, trading_days)
    max_dd = calculate_max_drawdown(port_daily_ret)
    calmar = calculate_calmar_ratio(ann_return, max_dd)

    var_95 = calculate_value_at_risk(port_daily_ret, 0.95)
    var_99 = calculate_value_at_risk(port_daily_ret, 0.99)
    cvar_95 = calculate_cvar(port_daily_ret, 0.95)

    div_metrics = calculate_diversification_metrics(weights)
    best_day, worst_day = calculate_extreme_returns(port_daily_ret)

    return {
        'expected_return': float(ann_return),
        'volatility': float(ann_vol),
        'variance': float(ann_var),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'max_drawdown': float(max_dd),
        'var_95': float(var_95),
        'var_99': float(var_99),
        'cvar_95': float(cvar_95),
        'hhi': float(div_metrics['hhi']),
        'effective_n': float(div_metrics['effective_n']),
        'best_day': float(best_day),
        'worst_day': float(worst_day)
    }
