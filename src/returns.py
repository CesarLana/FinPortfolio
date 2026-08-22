"""
Returns Module for FinPortfolio.
Calculates daily, cumulative, and annualized returns for individual assets and portfolios.
"""

from typing import Union
import pandas as pd
import numpy as np


def calculate_daily_returns(prices: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
    """
    Calculates daily returns of price series.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset closing prices.
    method : str, optional
        'arithmetic' for simple percentage change or 'log' for logarithmic returns.

    Returns
    -------
    pd.DataFrame
        Daily returns DataFrame with first NaN row dropped.
    """
    if prices.empty:
        raise ValueError("O DataFrame de preços está vazio.")

    if method == 'arithmetic':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("O método deve ser 'arithmetic' ou 'log'.")

    return returns


def calculate_cumulative_returns(daily_returns: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
    """
    Calculates cumulative returns over time.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame of daily returns.
    method : str, optional
        'arithmetic' or 'log'.

    Returns
    -------
    pd.DataFrame
        Cumulative returns series or DataFrame starting from 0.0.
    """
    if daily_returns.empty:
        raise ValueError("O DataFrame de retornos diários está vazio.")

    if method == 'arithmetic':
        cum_returns = (1 + daily_returns).cumprod() - 1
    elif method == 'log':
        cum_returns = np.exp(daily_returns.cumsum()) - 1
    else:
        raise ValueError("O método deve ser 'arithmetic' ou 'log'.")

    return cum_returns


def calculate_annualized_returns(
    daily_returns: pd.DataFrame,
    trading_days: int = 252,
    method: str = 'compounded'
) -> pd.Series:
    """
    Calculates expected annualized return for each asset.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame of daily returns.
    trading_days : int, optional
        Number of trading days per year (default 252).
    method : str, optional
        'compounded' for geometric mean annualized return: (1 + mean_daily)^trading_days - 1
        'arithmetic' for simple mean annualized return: mean_daily * trading_days

    Returns
    -------
    pd.Series
        Annualized expected return per asset.
    """
    if daily_returns.empty:
        raise ValueError("O DataFrame de retornos diários está vazio.")

    mean_daily = daily_returns.mean()

    if method == 'compounded':
        annualized = (1 + mean_daily) ** trading_days - 1
    elif method == 'arithmetic':
        annualized = mean_daily * trading_days
    else:
        raise ValueError("O método deve ser 'compounded' ou 'arithmetic'.")

    return annualized


def calculate_portfolio_daily_returns(
    daily_returns: pd.DataFrame,
    weights: Union[np.ndarray, list]
) -> pd.Series:
    """
    Calculates the weighted daily returns of a portfolio.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily returns of individual assets.
    weights : np.ndarray or list
        Array of portfolio weights summing to 1.

    Returns
    -------
    pd.Series
        Daily portfolio returns series.
    """
    w = np.array(weights)
    if not np.isclose(np.sum(w), 1.0, atol=1e-3):
        w = w / np.sum(w)  # Normalize weights if slight float precision drift

    portfolio_returns = daily_returns.dot(w)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns
