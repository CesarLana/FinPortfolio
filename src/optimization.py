"""
Optimization Module for FinPortfolio.
Implements Markowitz Modern Portfolio Theory (MPT) quadratic optimization using SciPy.
Calculates Global Minimum Variance (GMV), Maximum Sharpe Ratio (Tangency),
Target Return portfolios, and the Efficient Frontier curve.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.returns import calculate_annualized_returns
from src.risk import (
    calculate_covariance_matrix,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio
)


def _validate_bounds(
    num_assets: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Validates and formats min and max individual asset weight bounds.
    """
    if min_weight * num_assets > 1.0:
        min_weight = 1.0 / num_assets
    if max_weight * num_assets < 1.0:
        max_weight = 1.0

    min_w = max(0.0, min(min_weight, 1.0))
    max_w = min(1.0, max(max_weight, min_w))

    return [(min_w, max_w) for _ in range(num_assets)]


def optimize_gmv(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Finds the Global Minimum Variance (GMV) portfolio.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    min_weight : float, optional
        Minimum allocation weight per asset (default 0.0).
    max_weight : float, optional
        Maximum allocation weight per asset (default 1.0).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: 'weights', 'return', 'volatility', 'sharpe'
    """
    num_assets = len(expected_returns)
    init_weights = np.repeat(1.0 / num_assets, num_assets)
    bounds = _validate_bounds(num_assets, min_weight, max_weight)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def objective(w: np.ndarray) -> float:
        return np.dot(w.T, np.dot(cov_matrix.values, w))

    res = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    if not res.success:
        weights = init_weights
    else:
        weights = res.x / np.sum(res.x)

    ret = float(np.dot(weights, expected_returns.values))
    vol = calculate_portfolio_volatility(weights, cov_matrix)

    return {
        'weights': pd.Series(weights, index=expected_returns.index),
        'return': ret,
        'volatility': vol,
        'sharpe': calculate_sharpe_ratio(ret, vol)
    }


def optimize_max_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.1075,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Finds the Maximum Sharpe Ratio (Tangency) portfolio.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    risk_free_rate : float, optional
        Risk-free interest rate (default 0.1075).
    min_weight : float, optional
        Minimum allocation weight per asset.
    max_weight : float, optional
        Maximum allocation weight per asset.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: 'weights', 'return', 'volatility', 'sharpe'
    """
    num_assets = len(expected_returns)
    init_weights = np.repeat(1.0 / num_assets, num_assets)
    bounds = _validate_bounds(num_assets, min_weight, max_weight)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    def negative_sharpe(w: np.ndarray) -> float:
        port_ret = np.dot(w, expected_returns.values)
        port_vol = calculate_portfolio_volatility(w, cov_matrix)
        if port_vol <= 1e-8:
            return 0.0
        excess_return = port_ret - risk_free_rate
        # For negative excess returns, preserve correct directional ranking
        if excess_return < 0:
            return -excess_return * port_vol
        return -excess_return / port_vol

    res = minimize(
        negative_sharpe,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    if not res.success:
        weights = init_weights
    else:
        weights = res.x / np.sum(res.x)

    ret = float(np.dot(weights, expected_returns.values))
    vol = calculate_portfolio_volatility(weights, cov_matrix)
    sharpe = calculate_sharpe_ratio(ret, vol, risk_free_rate)

    return {
        'weights': pd.Series(weights, index=expected_returns.index),
        'return': ret,
        'volatility': vol,
        'sharpe': sharpe
    }


def optimize_target_return(
    target_return: float,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Minimizes volatility for a specified target return.

    Parameters
    ----------
    target_return : float
        Desired target expected return.
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    min_weight : float, optional
        Minimum weight per asset.
    max_weight : float, optional
        Maximum weight per asset.

    Returns
    -------
    Optional[Dict[str, Any]]
        Portfolio details or None if optimization fails to converge.
    """
    num_assets = len(expected_returns)
    init_weights = np.repeat(1.0 / num_assets, num_assets)
    bounds = _validate_bounds(num_assets, min_weight, max_weight)

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns.values) - target_return}
    )

    def objective(w: np.ndarray) -> float:
        return np.dot(w.T, np.dot(cov_matrix.values, w))

    res = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 500}
    )

    if not res.success:
        return None

    weights = res.x / np.sum(res.x)
    ret = float(np.dot(weights, expected_returns.values))
    vol = calculate_portfolio_volatility(weights, cov_matrix)

    return {
        'weights': pd.Series(weights, index=expected_returns.index),
        'return': ret,
        'volatility': vol,
        'sharpe': calculate_sharpe_ratio(ret, vol)
    }


def calculate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.1075
) -> pd.DataFrame:
    """
    Generates points along the Markowitz Efficient Frontier curve.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    num_points : int, optional
        Number of points along the frontier (default 50).
    min_weight : float, optional
        Minimum weight per asset.
    max_weight : float, optional
        Maximum weight per asset.
    risk_free_rate : float, optional
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['return', 'volatility', 'sharpe']
    """
    num_assets = len(expected_returns)
    bounds = _validate_bounds(num_assets, min_weight, max_weight)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # 1. Find GMV return (lower bound of efficient frontier)
    gmv_port = optimize_gmv(expected_returns, cov_matrix, min_weight, max_weight)
    min_ret = gmv_port['return']

    # 2. Find max achievable return under individual bounds
    max_ret_res = minimize(
        lambda w: -np.dot(w, expected_returns.values),
        np.repeat(1.0 / num_assets, num_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    max_ret = float(-max_ret_res.fun) if max_ret_res.success else expected_returns.max()

    if min_ret >= max_ret:
        target_returns = np.array([min_ret])
    else:
        target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier_points = []
    for tr in target_returns:
        res = optimize_target_return(tr, expected_returns, cov_matrix, min_weight, max_weight)
        if res is not None:
            sharpe = calculate_sharpe_ratio(res['return'], res['volatility'], risk_free_rate)
            frontier_points.append({
                'return': res['return'],
                'volatility': res['volatility'],
                'sharpe': sharpe
            })

    if not frontier_points:
        frontier_points.append({
            'return': gmv_port['return'],
            'volatility': gmv_port['volatility'],
            'sharpe': gmv_port['sharpe']
        })

    return pd.DataFrame(frontier_points)


def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series,
    risk_free_rate: float = 0.1075,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Finds the Risk Parity portfolio where each asset has equal marginal
    contribution to total portfolio risk (Equal Risk Contribution - ERC).

    Widely used by asset managers (Bridgewater All Weather, etc.) and banking
    allocation desks. Solves:

        min sum_i (w_i * (Cov * w)_i / sigma_p - sigma_p / N)^2

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    expected_returns : pd.Series
        Annualized expected returns per asset.
    risk_free_rate : float, optional
        Risk-free rate.
    min_weight : float, optional
        Minimum weight per asset.
    max_weight : float, optional
        Maximum weight per asset.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: 'weights', 'return', 'volatility', 'sharpe'
    """
    num_assets = len(cov_matrix)
    init_weights = np.repeat(1.0 / num_assets, num_assets)
    bounds = _validate_bounds(num_assets, min_weight, max_weight)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    cov_vals = cov_matrix.values
    target_risk_share = 1.0 / num_assets

    def risk_parity_objective(w: np.ndarray) -> float:
        port_var = np.dot(w.T, np.dot(cov_vals, w))
        port_vol = np.sqrt(max(1e-12, port_var))

        # Marginal risk contribution
        mrc = np.dot(cov_vals, w) / port_vol
        # Component risk contribution
        crc = w * mrc
        # Percentage risk contributions
        rc_pct = crc / port_vol

        # Objective: minimize deviation from equal risk contribution
        return float(np.sum((rc_pct - target_risk_share) ** 2))

    res = minimize(
        risk_parity_objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    if not res.success:
        weights = init_weights
    else:
        weights = res.x / np.sum(res.x)

    ret = float(np.dot(weights, expected_returns.values))
    vol = calculate_portfolio_volatility(weights, cov_matrix)
    sharpe = calculate_sharpe_ratio(ret, vol, risk_free_rate)

    return {
        'weights': pd.Series(weights, index=expected_returns.index),
        'return': ret,
        'volatility': vol,
        'sharpe': sharpe
    }
