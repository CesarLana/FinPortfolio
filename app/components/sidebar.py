"""
Sidebar component for Streamlit dashboard.
Collects user inputs for tickers, dates, risk-free rate, constraints, and simulations.
"""

from typing import Dict, Any, List
import datetime
import streamlit as st

# Preset portfolio configurations
PRESET_PORTFOLIOS = {
    "Personalizado": [],
    "B3 Blue Chips (Brasil)": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "BBAS3.SA"],
    "ETFs & Fundos de Índice": ["BOVA11.SA", "IVVB11.SA", "SMAL11.SA", "GOLD11.SA"],
    "Tech Giants (US)": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
    "FIIs Destaque": ["HGLG11.SA", "KNCR11.SA", "MXRF11.SA", "XPML11.SA"]
}


def render_sidebar() -> Dict[str, Any]:
    """
    Renders the sidebar controls and returns user parameter settings.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all configuration parameters.
    """
    st.sidebar.title("📊 Configurações da Carteira")
    st.sidebar.markdown("---")

    # 1. Preset Selector
    preset_choice = st.sidebar.selectbox(
        "🎯 Carteiras Pré-definidas",
        options=list(PRESET_PORTFOLIOS.keys()),
        index=1  # Default B3 Blue Chips
    )

    default_tickers = PRESET_PORTFOLIOS[preset_choice] if preset_choice != "Personalizado" else ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]

    # 2. Tickers Input
    selected_tickers = st.sidebar.multiselect(
        "🔤 Selecione os Ativos",
        options=list(set(default_tickers + ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BOVA11.SA", "IVVB11.SA", "AAPL", "MSFT", "NVDA", "BBDC4.SA", "RENT3.SA"])),
        default=default_tickers,
        help="Insira códigos válidos do Yahoo Finance (ex: PETR4.SA, AAPL, BOVA11.SA)."
    )

    # Custom text input to append extra tickers easily
    custom_ticker_str = st.sidebar.text_input(
        "➕ Adicionar Outro Ticker (opcional)",
        value="",
        placeholder="Ex: RADL3.SA, TSLA, B3SA3.SA",
        help="Separe múltiplos tickers por vírgula."
    )

    if custom_ticker_str.strip():
        extra_tickers = [t.strip().upper() for t in custom_ticker_str.split(",") if t.strip()]
        selected_tickers = list(dict.fromkeys(selected_tickers + extra_tickers))

    # Benchmark selector
    benchmark_ticker = st.sidebar.selectbox(
        "📈 Benchmark de Comparação",
        options=["BOVA11.SA", "IVVB11.SA", "SPY"],
        index=0,
        help="Ativo usado para comparar a performance histórica no Backtest."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Período da Análise")

    # Date preset quick selection
    period_option = st.sidebar.radio(
        "Intervalo Rápido",
        options=["1 Ano", "3 Anos", "5 Anos", "Personalizado"],
        index=1,
        horizontal=True
    )

    today = datetime.date.today()
    if period_option == "1 Ano":
        start_date = today - datetime.timedelta(days=365)
        end_date = today
    elif period_option == "3 Anos":
        start_date = today - datetime.timedelta(days=365 * 3)
        end_date = today
    elif period_option == "5 Anos":
        start_date = today - datetime.timedelta(days=365 * 5)
        end_date = today
    else:
        start_date = st.sidebar.date_input("Data Inicial", today - datetime.timedelta(days=365 * 3))
        end_date = st.sidebar.date_input("Data Final", today)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parâmetros de Otimização")

    # Risk-free rate input
    rf_pct = st.sidebar.number_input(
        "Taxa Livre de Risco Anual (%)",
        min_value=0.0,
        max_value=30.0,
        value=10.75,
        step=0.25,
        help="Taxa Selic ou CDI anual para cálculo do Índice de Sharpe."
    )
    risk_free_rate = rf_pct / 100.0

    # Asset exposure bounds
    col_min, col_max = st.sidebar.columns(2)
    with col_min:
        min_w_pct = st.number_input("Peso Mín por Ativo (%)", 0.0, 50.0, 0.0, step=1.0)
    with col_max:
        max_w_pct = st.number_input("Peso Máx por Ativo (%)", 10.0, 100.0, 100.0, step=5.0)

    min_weight = min_w_pct / 100.0
    max_weight = max_w_pct / 100.0

    # Monte Carlo simulations count
    num_simulations = st.sidebar.select_slider(
        "Qtd. de Carteiras Simuladas (Monte Carlo)",
        options=[1000, 3000, 5000, 10000, 20000],
        value=5000
    )

    initial_investment = st.sidebar.number_input(
        "Investimento Inicial Backtest (R$)",
        min_value=100.0,
        max_value=10000000.0,
        value=10000.0,
        step=1000.0
    )

    return {
        "tickers": selected_tickers,
        "benchmark": benchmark_ticker,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "risk_free_rate": risk_free_rate,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "num_simulations": num_simulations,
        "initial_investment": initial_investment
    }
