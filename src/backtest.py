"""
Backtest Module for FinPortfolio.
Simulates historical performance of optimized Markowitz portfolios
against benchmarks (e.g. BOVA11.SA) and Equal-Weighted (1/N) portfolios.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.returns import (
    calculate_portfolio_daily_returns,
    calculate_cumulative_returns
)
from src.risk import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_beta_and_alpha
)


def run_historical_backtest(
    daily_returns: pd.DataFrame,
    optimized_weights: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.1075,
    initial_investment: float = 10000.0,
    trading_days: int = 252
) -> Dict[str, Any]:
    """
    Runs a historical backtest comparing:
    1. Optimized MPT Portfolio
    2. Equal-Weighted Portfolio (1/N)
    3. Benchmark Asset (if provided, e.g. BOVA11.SA)

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Asset daily returns.
    optimized_weights : pd.Series
        Weights of the optimized MPT portfolio.
    benchmark_returns : Optional[pd.Series], optional
        Daily returns series of benchmark ticker.
    risk_free_rate : float, optional
        Annualized risk-free rate (default 0.1075).
    initial_investment : float, optional
        Starting capital for equity curve visualization (default 10,000.0).
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'wealth_curves': pd.DataFrame of cumulative monetary values
        - 'cumulative_returns': pd.DataFrame of cumulative percentage returns
        - 'drawdowns': pd.DataFrame of drawdowns over time
        - 'metrics': pd.DataFrame of summary performance metrics per strategy
        - 'beta_alpha': Dict containing Beta and Alpha of strategies against benchmark
    """
    num_assets = len(daily_returns.columns)
    eq_weights = np.repeat(1.0 / num_assets, num_assets)

    # Calculate strategy daily returns
    opt_daily = calculate_portfolio_daily_returns(daily_returns, optimized_weights.values)
    opt_daily.name = "Carteira Otimizada"

    eq_daily = calculate_portfolio_daily_returns(daily_returns, eq_weights)
    eq_daily.name = "Carteira 1/N (Igual)"

    strategies_daily = pd.DataFrame({
        "Carteira Otimizada": opt_daily,
        "Carteira 1/N (Igual)": eq_daily
    })

    has_bench = False
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align index responsibly
        bench = benchmark_returns.reindex(daily_returns.index).ffill().fillna(0.0)
        bench.name = "Benchmark"
        strategies_daily["Benchmark"] = bench
        has_bench = True

    # Cumulative returns & Wealth curves
    cum_returns = (1 + strategies_daily).cumprod() - 1
    wealth_curves = initial_investment * (1 + cum_returns)

    # Compute Drawdown series for each strategy
    drawdowns = pd.DataFrame(index=strategies_daily.index)
    for col in strategies_daily.columns:
        wealth = wealth_curves[col]
        peak = wealth.cummax()
        drawdowns[col] = (wealth - peak) / peak

    # Calculate summary metrics per strategy
    metrics_list = []
    num_days = len(daily_returns)

    for col in strategies_daily.columns:
        ret_series = strategies_daily[col]
        cum_ret = cum_returns[col].iloc[-1]

        # CAGR
        if num_days > 0:
            cagr = (1 + cum_ret) ** (trading_days / num_days) - 1
        else:
            cagr = 0.0

        _, ann_vol = calculate_volatility(ret_series.to_frame(), trading_days)
        vol = ann_vol.iloc[0]
        sharpe = calculate_sharpe_ratio(cagr, vol, risk_free_rate)
        sortino = calculate_sortino_ratio(ret_series, cagr, risk_free_rate, trading_days)
        mdd = calculate_max_drawdown(ret_series)
        calmar = calculate_calmar_ratio(cagr, mdd)

        item = {
            'Estratégia': col,
            'Retorno Total (%)': cum_ret * 100,
            'Retorno Anualizado (CAGR %)': cagr * 100,
            'Volatilidade Anualizada (%)': vol * 100,
            'Índice de Sharpe': sharpe,
            'Índice de Sortino': sortino,
            'Índice de Calmar': calmar,
            'Maximum Drawdown (%)': mdd * 100
        }

        if has_bench and col != "Benchmark":
            beta, alpha = calculate_beta_and_alpha(ret_series, strategies_daily["Benchmark"], risk_free_rate, trading_days)
            item['Beta (vs Bench)'] = beta
            item['Alpha de Jensen (%)'] = alpha * 100
        elif has_bench and col == "Benchmark":
            item['Beta (vs Bench)'] = 1.0
            item['Alpha de Jensen (%)'] = 0.0

        metrics_list.append(item)

    metrics_df = pd.DataFrame(metrics_list).set_index('Estratégia')

    return {
        'wealth_curves': wealth_curves,
        'cumulative_returns': cum_returns,
        'drawdowns': drawdowns,
        'metrics': metrics_df
    }
