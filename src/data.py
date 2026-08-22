"""
Data Module for FinPortfolio.
Responsible for fetching historical market prices, cleaning missing values,
and ensuring data integrity without synthetic data generation.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


def fetch_historical_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    min_observations: int = 30
) -> pd.DataFrame:
    """
    Downloads historical close price data for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of asset ticker symbols (e.g. ['PETR4.SA', 'VALE3.SA', 'AAPL']).
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    min_observations : int, optional
        Minimum required trading days for an asset to be retained, default 30.

    Returns
    -------
    pd.DataFrame
        DataFrame of cleaned closing prices with DatetimeIndex and asset tickers as columns.

    Raises
    ------
    ValueError
        If input tickers is empty, date range is invalid, or no valid ticker data could be fetched.
    """
    if not tickers:
        raise ValueError("A lista de tickers não pode estar vazia.")

    # Clean ticker string formatting
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean_tickers:
        raise ValueError("Nenhum ticker válido foi fornecido.")

    # Remove duplicate tickers while preserving order
    clean_tickers = list(dict.fromkeys(clean_tickers))

    logger.info(f"Downloading data for {clean_tickers} from {start_date} to {end_date}")

    try:
        df_raw = yf.download(
            tickers=clean_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar dados do Yahoo Finance: {str(e)}")

    if df_raw.empty:
        raise ValueError(
            f"Nenhum dado encontrado para os tickers {clean_tickers} no período de {start_date} a {end_date}."
        )

    # Extract Close prices (yfinance output format handling)
    if isinstance(df_raw.columns, pd.MultiIndex):
        # MultiIndex columns e.g. ('Close', 'PETR4.SA')
        if 'Close' in df_raw.columns.levels[0]:
            prices = df_raw['Close'].copy()
        elif 'Close' in df_raw.columns.levels[1]:
            prices = df_raw.xs('Close', axis=1, level=1, drop_level=True).copy()
        else:
            prices = df_raw.copy()
    else:
        # Single ticker download or simple columns
        if 'Close' in df_raw.columns:
            prices = df_raw[['Close']].copy()
            prices.columns = clean_tickers[:1]
        else:
            prices = df_raw.copy()

    # Convert to DataFrame if Series (e.g. single ticker)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=clean_tickers[0])

    # Ensure index is datetime and strip timezones if present
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    # Remove duplicate dates in index
    prices = prices[~prices.index.duplicated(keep='first')]
    prices.sort_index(inplace=True)

    # Filter out columns with excessive missing data (> 20%)
    valid_cols = []
    for col in prices.columns:
        null_pct = prices[col].isnull().mean()
        if null_pct > 0.20:
            logger.warning(f"Ticker {col} descartado por ter {null_pct:.1%} de dados ausentes.")
        elif len(prices[col].dropna()) < min_observations:
            logger.warning(f"Ticker {col} descartado por ter menos de {min_observations} observações válidas.")
        else:
            valid_cols.append(col)

    if not valid_cols:
        raise ValueError("Nenhum dos tickers possui histórico suficiente de dados no período selecionado.")

    prices = prices[valid_cols]

    # Clean missing values responsibly: forward fill then backward fill
    prices = prices.ffill().bfill()

    # Drop any remaining rows with NaN
    prices.dropna(inplace=True)

    if len(prices) < min_observations:
        raise ValueError(
            f"Período selecionado resultou em apenas {len(prices)} dias úteis, "
            f"insuficiente para análise (mínimo: {min_observations})."
        )

    return prices


def normalize_prices(prices: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    """
    Normalizes asset prices to a base value (default 100) for comparative visualization.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.
    base_value : float, optional
        Base value for normalization, default 100.0.

    Returns
    -------
    pd.DataFrame
        Normalized prices DataFrame.
    """
    if prices.empty:
        return prices
    return (prices / prices.iloc[0]) * base_value
