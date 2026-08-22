"""
Stress Testing & Scenario Analysis Module for FinPortfolio.
Simulates portfolio behavior under historical crisis events and custom shocks.
Essential for banking risk management (Basel III/IV), ALM desks, and regulatory reporting.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.returns import calculate_portfolio_daily_returns
from src.risk import calculate_max_drawdown


# Historical crisis periods for Brazilian and global markets
HISTORICAL_SCENARIOS = {
    "Crise Financeira Global (2008)": {
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "Colapso do Lehman Brothers e crise do subprime. Ibovespa caiu ~60% do topo."
    },
    "Joesley Day (2017)": {
        "start": "2017-05-17",
        "end": "2017-06-30",
        "description": "Delação JBS. Circuit breaker na B3 com queda de 8.8% em um único dia."
    },
    "COVID-19 Crash (2020)": {
        "start": "2020-02-20",
        "end": "2020-04-30",
        "description": "Pandemia global. Circuit breakers consecutivos, queda acumulada ~46% no Ibovespa."
    },
    "Greve dos Caminhoneiros (2018)": {
        "start": "2018-05-21",
        "end": "2018-06-29",
        "description": "Paralisação nacional com impacto em logística e combustível."
    },
    "Taper Tantrum Emergentes (2013)": {
        "start": "2013-05-22",
        "end": "2013-08-31",
        "description": "Fed sinalizou redução de estímulos. Fuga de capital de emergentes."
    },
    "Crise Fiscal e Downgrade BR (2015)": {
        "start": "2015-07-01",
        "end": "2016-01-31",
        "description": "Perda do grau de investimento pela S&P. Recessão e crise política."
    }
}


def run_historical_stress_test(
    daily_returns: pd.DataFrame,
    weights: pd.Series,
    initial_investment: float = 10000.0
) -> Dict[str, Any]:
    """
    Runs the portfolio through historical crisis periods that overlap with available data.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Full daily returns DataFrame.
    weights : pd.Series
        Portfolio weights.
    initial_investment : float, optional
        Starting capital for impact calculation.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'summary' (DataFrame of scenario impacts) and
        'details' (Dict of per-scenario daily return series).
    """
    port_daily = calculate_portfolio_daily_returns(daily_returns, weights.values)
    data_start = daily_returns.index.min()
    data_end = daily_returns.index.max()

    results = []
    details = {}

    for scenario_name, info in HISTORICAL_SCENARIOS.items():
        s_start = pd.Timestamp(info["start"])
        s_end = pd.Timestamp(info["end"])

        # Check if scenario overlaps with available data
        if s_start > data_end or s_end < data_start:
            continue

        # Clip to available data range
        effective_start = max(s_start, data_start)
        effective_end = min(s_end, data_end)

        mask = (port_daily.index >= effective_start) & (port_daily.index <= effective_end)
        scenario_returns = port_daily[mask]

        if len(scenario_returns) < 2:
            continue

        cum_ret = (1 + scenario_returns).cumprod().iloc[-1] - 1
        max_dd = calculate_max_drawdown(scenario_returns)
        worst_day = float(scenario_returns.min())
        best_day = float(scenario_returns.max())
        vol = float(scenario_returns.std() * np.sqrt(252))
        loss_brl = initial_investment * cum_ret
        trading_days = len(scenario_returns)

        results.append({
            "Cenário": scenario_name,
            "Período": f"{effective_start.strftime('%d/%m/%Y')} - {effective_end.strftime('%d/%m/%Y')}",
            "Dias Úteis": trading_days,
            "Retorno Acumulado (%)": cum_ret * 100,
            "Max Drawdown (%)": max_dd * 100,
            "Pior Dia (%)": worst_day * 100,
            "Melhor Dia (%)": best_day * 100,
            "Vol. Anualizada (%)": vol * 100,
            "Impacto (R$)": loss_brl,
            "Descrição": info["description"]
        })

        details[scenario_name] = scenario_returns

    summary_df = pd.DataFrame(results) if results else pd.DataFrame()

    return {
        "summary": summary_df,
        "details": details
    }


def run_custom_shock(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: pd.Series,
    shock_map: Dict[str, float],
    initial_investment: float = 10000.0
) -> Dict[str, Any]:
    """
    Applies a user-defined percentage shock to each asset and estimates portfolio impact.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    weights : pd.Series
        Portfolio weights.
    shock_map : Dict[str, float]
        Dictionary mapping ticker -> shock percentage (e.g. {"PETR4.SA": -0.20} for -20%).
    initial_investment : float, optional
        Starting capital.

    Returns
    -------
    Dict[str, Any]
        Dictionary with portfolio-level shock impact.
    """
    w = weights.values
    shocks = np.array([shock_map.get(t, 0.0) for t in weights.index])

    portfolio_shock = float(np.dot(w, shocks))
    loss_brl = initial_investment * portfolio_shock

    per_asset_contrib = []
    for i, ticker in enumerate(weights.index):
        s = shock_map.get(ticker, 0.0)
        contrib = w[i] * s
        per_asset_contrib.append({
            "Ativo": ticker,
            "Peso (%)": w[i] * 100,
            "Choque Aplicado (%)": s * 100,
            "Contribuição ao Impacto (%)": contrib * 100
        })

    return {
        "portfolio_shock_pct": portfolio_shock * 100,
        "portfolio_loss_brl": loss_brl,
        "per_asset": pd.DataFrame(per_asset_contrib)
    }


def calculate_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Decomposes total portfolio risk into Marginal Risk Contribution (MRC) and
    Component Risk Contribution (CRC) per asset.

    Used by risk desks to understand which assets are driving portfolio volatility.

    MRC_i = (Cov * w)_i / sigma_p
    CRC_i = w_i * MRC_i
    % Risk Contribution_i = CRC_i / sigma_p

    Parameters
    ----------
    weights : pd.Series
        Portfolio asset weights.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Ativo, Peso (%), MRC, CRC, Contribuição ao Risco (%).
    """
    w = weights.values
    cov = cov_matrix.values
    port_var = np.dot(w.T, np.dot(cov, w))
    port_vol = np.sqrt(max(0.0, port_var))

    if port_vol < 1e-8:
        n = len(w)
        return pd.DataFrame({
            "Ativo": weights.index,
            "Peso (%)": w * 100,
            "MRC (Marginal Risk)": np.zeros(n),
            "CRC (Component Risk)": np.zeros(n),
            "Contribuição ao Risco (%)": np.full(n, 100.0 / n)
        })

    # Marginal Risk Contribution: derivative of portfolio vol w.r.t. weight_i
    mrc = np.dot(cov, w) / port_vol

    # Component Risk Contribution: w_i * MRC_i
    crc = w * mrc

    # Percentage risk contribution
    pct_contrib = crc / port_vol * 100

    return pd.DataFrame({
        "Ativo": weights.index,
        "Peso (%)": np.round(w * 100, 2),
        "MRC (Marginal Risk)": np.round(mrc * 100, 4),
        "CRC (Component Risk)": np.round(crc * 100, 4),
        "Contribuição ao Risco (%)": np.round(pct_contrib, 2)
    })


def calculate_rolling_metrics(
    daily_returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.1075,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates rolling volatility, rolling Sharpe, and rolling cumulative return.

    Parameters
    ----------
    daily_returns : pd.Series
        Portfolio daily returns.
    window : int, optional
        Rolling window in trading days (default 63 = ~3 months).
    risk_free_rate : float, optional
        Annualized risk-free rate.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: rolling_vol, rolling_sharpe, rolling_return.
    """
    rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(trading_days)
    rolling_mean = daily_returns.rolling(window=window).mean() * trading_days
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol

    rolling_cum = (1 + daily_returns).rolling(window=window).apply(
        lambda x: x.prod() - 1, raw=True
    )

    return pd.DataFrame({
        "rolling_vol": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
        "rolling_return": rolling_cum
    }).dropna()


def calculate_tracking_error_and_ir(
    portfolio_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculates Tracking Error and Information Ratio relative to a benchmark.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Portfolio daily returns.
    benchmark_daily_returns : pd.Series
        Benchmark daily returns.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'tracking_error' (annualized) and 'information_ratio'.
    """
    aligned = pd.concat([portfolio_daily_returns, benchmark_daily_returns], axis=1).dropna()
    if len(aligned) < 20:
        return {"tracking_error": 0.0, "information_ratio": 0.0}

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active_returns.std() * np.sqrt(trading_days))
    mean_active = float(active_returns.mean() * trading_days)

    ir = mean_active / te if te > 1e-8 else 0.0

    return {
        "tracking_error": te,
        "information_ratio": ir
    }
