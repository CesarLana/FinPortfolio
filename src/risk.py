"""
Risk Module for FinPortfolio.
Calculates volatility, variance, covariance, correlation, Sharpe ratio, Sortino ratio,
Calmar ratio, Beta, Alpha, Maximum Drawdown, Value at Risk (VaR), Conditional VaR (CVaR),
and diversification metrics (HHI & ENC).
"""

from typing import Tuple, Union, Dict
import pandas as pd
import numpy as np


def calculate_volatility(
    daily_returns: pd.DataFrame,
    trading_days: int = 252
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates daily and annualized volatility for assets.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame of asset daily returns.
    trading_days : int, optional
        Number of trading days per year (default 252).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (Daily Volatility, Annualized Volatility)
    """
    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(trading_days)
    return daily_vol, annualized_vol


def calculate_variance(daily_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Calculates annualized variance for assets.
    """
    _, ann_vol = calculate_volatility(daily_returns, trading_days)
    return ann_vol ** 2


def calculate_covariance_matrix(
    daily_returns: pd.DataFrame,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates annualized covariance matrix.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily returns DataFrame.
    trading_days : int, optional
        Trading days per year (default 252).

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix (N x N).
    """
    return daily_returns.cov() * trading_days


def calculate_correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates asset correlation matrix (-1 to +1).
    """
    return daily_returns.corr()


def calculate_portfolio_volatility(
    weights: Union[np.ndarray, list],
    cov_matrix: pd.DataFrame
) -> float:
    """
    Calculates portfolio annualized volatility given weights and covariance matrix.

    sigma_p = sqrt(w^T * Cov * w)

    Parameters
    ----------
    weights : np.ndarray or list
        Asset weights array.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    float
        Portfolio annualized volatility.
    """
    w = np.array(weights)
    variance = np.dot(w.T, np.dot(cov_matrix.values, w))
    return float(np.sqrt(np.maximum(0.0, variance)))


def calculate_sharpe_ratio(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.1075
) -> float:
    """
    Calculates Sharpe Ratio: (E[R_p] - R_f) / sigma_p.

    Parameters
    ----------
    expected_return : float
        Annualized expected portfolio return.
    volatility : float
        Annualized portfolio volatility.
    risk_free_rate : float, optional
        Annualized risk-free interest rate (e.g. 0.1075 for 10.75%).

    Returns
    -------
    float
        Sharpe Ratio.
    """
    if volatility <= 0 or np.isnan(volatility):
        return 0.0
    return (expected_return - risk_free_rate) / volatility


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    expected_return: float,
    risk_free_rate: float = 0.1075,
    trading_days: int = 252
) -> float:
    """
    Calculates Sortino Ratio: (E[R_p] - R_f) / Downside_Deviation.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns series.
    expected_return : float
        Annualized expected return.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    float
        Sortino Ratio.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    negative_excess = np.minimum(0.0, daily_returns - daily_rf)
    downside_std = np.sqrt(np.mean(negative_excess ** 2)) * np.sqrt(trading_days)

    if downside_std <= 1e-8 or np.isnan(downside_std):
        return 0.0
    return (expected_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float
) -> float:
    """
    Calculates Calmar Ratio: CAGR / abs(Max_Drawdown).

    Parameters
    ----------
    cagr : float
        Compound Annual Growth Rate.
    max_drawdown : float
        Maximum Drawdown (negative float).

    Returns
    -------
    float
        Calmar Ratio.
    """
    abs_mdd = abs(max_drawdown)
    if abs_mdd <= 1e-6:
        return 0.0
    return cagr / abs_mdd


def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Calculates Maximum Drawdown (MDD) from daily returns or cumulative prices.

    Parameters
    ----------
    series : pd.Series
        Can be daily returns or cumulative prices.

    Returns
    -------
    float
        Maximum Drawdown expressed as a negative float (e.g. -0.25 = -25%).
    """
    if series.empty:
        return 0.0

    # If series is daily returns, convert to wealth index first
    if series.min() < 0 and series.max() < 2.0 and not (series > 1.0).any():
        wealth_index = (1 + series).cumprod()
    else:
        wealth_index = series

    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return float(drawdowns.min())


def calculate_value_at_risk(
    daily_returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculates Historical Value at Risk (VaR) for a given confidence level.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns.
    confidence_level : float, optional
        Confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        Daily VaR percentage (negative float, e.g. -0.021 for -2.1%).
    """
    if daily_returns.empty:
        return 0.0
    percentile = (1.0 - confidence_level) * 100
    var = float(np.percentile(daily_returns.dropna(), percentile))
    return var


def calculate_cvar(
    daily_returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculates Conditional Value at Risk (CVaR / Expected Shortfall).
    Measures the average loss occurring in the worst (1 - confidence_level)% days.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns.
    confidence_level : float, optional
        Confidence level, default 0.95.

    Returns
    -------
    float
        CVaR percentage (negative float).
    """
    if daily_returns.empty:
        return 0.0
    var = calculate_value_at_risk(daily_returns, confidence_level)
    tail_losses = daily_returns[daily_returns <= var]
    if tail_losses.empty:
        return var
    return float(tail_losses.mean())


def calculate_beta_and_alpha(
    portfolio_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    risk_free_rate: float = 0.1075,
    trading_days: int = 252
) -> Tuple[float, float]:
    """
    Calculates Portfolio Beta (systematic risk) and Jensen's Alpha against a Benchmark.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Daily portfolio returns.
    benchmark_daily_returns : pd.Series
        Daily benchmark returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    Tuple[float, float]
        (Beta, Annualized Alpha)
    """
    aligned = pd.concat([portfolio_daily_returns, benchmark_daily_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0, 0.0

    p_ret = aligned.iloc[:, 0]
    b_ret = aligned.iloc[:, 1]

    cov = np.cov(p_ret, b_ret)[0, 1]
    var_b = np.var(b_ret)

    if var_b <= 1e-8:
        return 1.0, 0.0

    beta = float(cov / var_b)

    # Annualized Alpha (Jensen's Alpha)
    p_cagr = (1 + p_ret.mean()) ** trading_days - 1
    b_cagr = (1 + b_ret.mean()) ** trading_days - 1
    alpha = float(p_cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate)))

    return beta, alpha


def calculate_diversification_metrics(weights: Union[np.ndarray, list, pd.Series]) -> Dict[str, float]:
    """
    Calculates portfolio concentration and diversification metrics:
    - Herfindahl-Hirschman Index (HHI)
    - Effective Number of Constituents (ENC = 1 / HHI)

    Parameters
    ----------
    weights : Union[np.ndarray, list, pd.Series]
        Asset weight vector.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'hhi' and 'effective_n'.
    """
    w = np.array(weights)
    w_clean = w[w > 1e-6]
    hhi = float(np.sum(w_clean ** 2))
    enc = float(1.0 / hhi) if hhi > 0 else 0.0

    return {
        'hhi': hhi,
        'effective_n': enc
    }


def calculate_extreme_returns(daily_returns: pd.Series) -> Tuple[float, float]:
    """
    Returns the best single-day return and worst single-day return.

    Returns
    -------
    Tuple[float, float]
        (best_day_return, worst_day_return)
    """
    if daily_returns.empty:
        return 0.0, 0.0
    return float(daily_returns.max()), float(daily_returns.min())
