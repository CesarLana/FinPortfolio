"""
FinPortfolio - Main Streamlit Dashboard Application.
Combines Markowitz MPT optimization, Monte Carlo simulations, risk metrics,
historical backtesting, stress testing, risk contribution, and rolling analytics.
"""

import sys
import os
from pathlib import Path

# Add project root directory to python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np

from src.data import fetch_historical_data, normalize_prices
from src.returns import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_annualized_returns,
    calculate_portfolio_daily_returns
)
from src.risk import (
    calculate_volatility,
    calculate_covariance_matrix,
    calculate_correlation_matrix,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_extreme_returns,
    calculate_diversification_metrics
)
from src.optimization import (
    optimize_gmv,
    optimize_max_sharpe,
    optimize_risk_parity,
    calculate_efficient_frontier
)
from src.portfolio import (
    simulate_random_portfolios,
    calculate_portfolio_metrics_summary
)
from src.backtest import run_historical_backtest
from src.stress_testing import (
    run_historical_stress_test,
    run_custom_shock,
    calculate_risk_contribution,
    calculate_rolling_metrics,
    calculate_tracking_error_and_ir
)

from app.components.sidebar import render_sidebar
from app.components.charts import (
    plot_normalized_prices,
    plot_correlation_heatmap,
    plot_efficient_frontier_and_simulations,
    plot_portfolio_weights_comparison,
    plot_portfolio_donut,
    plot_backtest_wealth_curves,
    plot_backtest_drawdowns,
    plot_returns_distribution_with_var,
    plot_risk_contribution,
    plot_rolling_metrics,
    plot_stress_test_waterfall,
    plot_multi_portfolio_comparison
)
from app.components.metrics import (
    render_metric_cards,
    render_regulatory_disclaimer,
    render_mpt_theory_explanation
)

# Set Streamlit Page Config
st.set_page_config(
    page_title="FinPortfolio - Otimizador de Carteiras Markowitz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished dark theme UI
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #29B6F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161B22;
        border-radius: 8px 8px 0px 0px;
        padding-left: 14px;
        padding-right: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        border-bottom: 3px solid #29B6F6 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(tickers: list, benchmark: str, start_date: str, end_date: str):
    """Cached function to download asset prices and benchmark prices."""
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    prices = fetch_historical_data(all_tickers, start_date, end_date)

    asset_prices = prices[[t for t in tickers if t in prices.columns]]
    benchmark_prices = prices[benchmark] if benchmark in prices.columns else None

    return asset_prices, benchmark_prices


def main():
    # Render Header Title
    st.title("📈 FinPortfolio — Análise e Otimização de Carteiras de Investimentos")
    st.caption("Sistema quantitativo profissional de otimização de portfólios baseado na Teoria Moderna de Portfólios (MPT) de Harry Markowitz.")

    # Render Sidebar and retrieve user settings
    params = render_sidebar()

    tickers = params["tickers"]
    benchmark = params["benchmark"]
    start_date = params["start_date"]
    end_date = params["end_date"]
    risk_free_rate = params["risk_free_rate"]
    min_weight = params["min_weight"]
    max_weight = params["max_weight"]
    num_simulations = params["num_simulations"]
    initial_investment = params["initial_investment"]

    if not tickers or len(tickers) < 2:
        st.info("👈 Por favor, selecione pelo menos **2 ativos** na barra lateral para realizar a otimização de carteira.")
        render_regulatory_disclaimer()
        return

    # Data Collection with Spinner
    with st.spinner("📥 Baixando e tratando dados históricos de mercado..."):
        try:
            asset_prices, benchmark_prices = load_data(tickers, benchmark, start_date, end_date)
        except Exception as e:
            st.error(f"❌ Erro ao obter dados de mercado: {str(e)}")
            st.warning("Dica: Verifique se os códigos dos tickers estão corretos e se há dados no período selecionado.")
            return

    if asset_prices.empty or len(asset_prices.columns) < 2:
        st.error("Não foram encontrados dados suficientes para pelo menos 2 dos ativos selecionados.")
        return

    # Calculate returns and risk matrices
    daily_returns = calculate_daily_returns(asset_prices)
    annualized_returns = calculate_annualized_returns(daily_returns)
    daily_vol, ann_vol = calculate_volatility(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)
    corr_matrix = calculate_correlation_matrix(daily_returns)

    # Perform Markowitz Optimization
    with st.spinner("⚡ Executando otimização quadrática e simulações Monte Carlo..."):
        gmv_port = optimize_gmv(annualized_returns, cov_matrix, min_weight, max_weight)
        max_sharpe_port = optimize_max_sharpe(annualized_returns, cov_matrix, risk_free_rate, min_weight, max_weight)
        risk_parity_port = optimize_risk_parity(cov_matrix, annualized_returns, risk_free_rate, min_weight, max_weight)
        frontier_df = calculate_efficient_frontier(annualized_returns, cov_matrix, 50, min_weight, max_weight, risk_free_rate)
        simulations_df, sim_weights = simulate_random_portfolios(
            annualized_returns, cov_matrix, num_simulations, risk_free_rate, min_weight, max_weight
        )

    # Calculate full metrics for Max Sharpe portfolio
    max_sharpe_metrics = calculate_portfolio_metrics_summary(
        max_sharpe_port["weights"], daily_returns, risk_free_rate
    )

    # Top KPI Metrics Display
    st.markdown("### 🏆 Performance da Carteira Otimizada (Máximo Sharpe)")
    render_metric_cards(max_sharpe_metrics)

    st.markdown("---")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visão Geral",
        "🎯 Fronteira Eficiente",
        "🛡️ Risco & VaR",
        "⏳ Backtest",
        "🏦 Institucional / Banking",
        "📖 Metodologia"
    ])

    # TAB 1: Visão Geral & Ativos
    with tab1:
        st.subheader("Evolução Histórica dos Preços Normalizados")
        norm_prices = normalize_prices(asset_prices)
        st.plotly_chart(plot_normalized_prices(norm_prices), use_container_width=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Resumo de Retorno e Risco Individual")
            asset_stats = pd.DataFrame({
                "Retorno Esperado (CAGR)": (annualized_returns * 100).map("{:.2f}%".format),
                "Volatilidade Anualizada": (ann_vol * 100).map("{:.2f}%".format),
                "Variância Anualizada": ((ann_vol ** 2) * 100).map("{:.4f}".format),
                "Sharpe Individual": ((annualized_returns - risk_free_rate) / ann_vol).map("{:.2f}".format)
            })
            st.dataframe(asset_stats, use_container_width=True)

        with col_right:
            st.subheader("Matriz de Correlação")
            st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)
            corr_csv = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Baixar Matriz de Correlação (CSV)",
                data=corr_csv,
                file_name="matriz_correlacao.csv",
                mime="text/csv"
            )

    # TAB 2: Fronteira Eficiente & Otimização
    with tab2:
        st.subheader("Gráfico da Fronteira Eficiente & Carteiras Simuladas")
        st.plotly_chart(
            plot_efficient_frontier_and_simulations(simulations_df, frontier_df, gmv_port, max_sharpe_port, risk_free_rate),
            use_container_width=True
        )

        st.markdown("---")
        col_bar, col_pie = st.columns([1.2, 1])

        with col_bar:
            st.subheader("Comparativo de Alocação de Pesos")
            st.plotly_chart(
                plot_portfolio_weights_comparison(gmv_port["weights"], max_sharpe_port["weights"]),
                use_container_width=True
            )

        with col_pie:
            st.subheader("Composição da Carteira Máximo Sharpe")
            st.plotly_chart(
                plot_portfolio_donut(max_sharpe_port["weights"], title="Alocação Máximo Sharpe"),
                use_container_width=True
            )

        st.subheader("Tabela de Pesos Otimizados")
        weights_df = pd.DataFrame({
            "Ativo": max_sharpe_port["weights"].index,
            "Máximo Sharpe (%)": (max_sharpe_port["weights"].values * 100).round(2),
            "Mínima Volatilidade (%)": (gmv_port["weights"].values * 100).round(2),
            "Risk Parity (%)": (risk_parity_port["weights"].values * 100).round(2)
        }).set_index("Ativo")

        st.dataframe(weights_df, use_container_width=True)

        csv_data = weights_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Baixar Pesos Otimizados (CSV)",
            data=csv_data,
            file_name="alocacao_pesos_markowitz.csv",
            mime="text/csv"
        )

    # TAB 3: Análise de Risco & VaR
    with tab3:
        st.subheader("Métricas Avançadas de Risco da Carteira Otimizada")

        port_daily_ret = calculate_portfolio_daily_returns(daily_returns, max_sharpe_port["weights"].values)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.info(f"**Value at Risk (VaR 95%)**: {max_sharpe_metrics['var_95']*100:.2f}%\n\nEm 95% dos dias, a perda máxima diária estimada não excede este valor.")
        with col_r2:
            st.error(f"**Value at Risk (VaR 99%)**: {max_sharpe_metrics['var_99']*100:.2f}%\n\nEm 99% dos dias, a perda máxima diária estimada não excede este valor.")
        with col_r3:
            st.warning(f"**Conditional VaR (CVaR 95%)**: {max_sharpe_metrics['cvar_95']*100:.2f}%\n\nPerda média nos 5% piores dias do histórico.")
        with col_r4:
            st.metric(label="Maximum Drawdown", value=f"{max_sharpe_metrics['max_drawdown']*100:.2f}%")

        st.markdown("---")
        col_dist, col_ext = st.columns([1.5, 1])
        with col_dist:
            st.plotly_chart(
                plot_returns_distribution_with_var(
                    port_daily_ret,
                    max_sharpe_metrics['var_95'],
                    max_sharpe_metrics['var_99'],
                    max_sharpe_metrics['cvar_95']
                ),
                use_container_width=True
            )
        with col_ext:
            st.subheader("Extremos Diários e Momentos Estatísticos")
            st.write(f"🟢 **Melhor Retorno Diário**: {max_sharpe_metrics['best_day']*100:.2f}%")
            st.write(f"🔴 **Pior Retorno Diário**: {max_sharpe_metrics['worst_day']*100:.2f}%")
            st.write(f"📊 **Volatilidade Diária**: {port_daily_ret.std()*100:.2f}%")
            st.write(f"📐 **Assimetria (Skewness)**: {port_daily_ret.skew():.2f}")
            st.write(f"📐 **Curtose (Kurtosis)**: {port_daily_ret.kurtosis():.2f}")
            st.write(f"🌐 **Índice de Herfindahl (HHI)**: {max_sharpe_metrics['hhi']:.4f}")
            st.write(f"🌐 **Ativos Efetivos (ENC)**: {max_sharpe_metrics['effective_n']:.2f}")

    # TAB 4: Backtest Histórico
    with tab4:
        st.subheader(f"Simulação de Backtest Histórico (Investimento Inicial: R$ {initial_investment:,.2f})")

        bench_returns = calculate_daily_returns(benchmark_prices.to_frame())[benchmark] if benchmark_prices is not None else None

        backtest_results = run_historical_backtest(
            daily_returns=daily_returns,
            optimized_weights=max_sharpe_port["weights"],
            benchmark_returns=bench_returns,
            risk_free_rate=risk_free_rate,
            initial_investment=initial_investment
        )

        st.plotly_chart(plot_backtest_wealth_curves(backtest_results["wealth_curves"]), use_container_width=True)
        st.plotly_chart(plot_backtest_drawdowns(backtest_results["drawdowns"]), use_container_width=True)

        st.subheader("Tabela Comparativa de Performance no Backtest")
        formatted_metrics = backtest_results["metrics"].copy()

        for col in ["Retorno Total (%)", "Retorno Anualizado (CAGR %)", "Volatilidade Anualizada (%)", "Maximum Drawdown (%)"]:
            if col in formatted_metrics.columns:
                formatted_metrics[col] = formatted_metrics[col].map("{:.2f}%".format)

        for col in ["Índice de Sharpe", "Índice de Sortino", "Índice de Calmar", "Beta (vs Bench)"]:
            if col in formatted_metrics.columns:
                formatted_metrics[col] = formatted_metrics[col].map("{:.2f}".format)

        if "Alpha de Jensen (%)" in formatted_metrics.columns:
            formatted_metrics["Alpha de Jensen (%)"] = formatted_metrics["Alpha de Jensen (%)"].map("{:.2f}%".format)

        st.dataframe(formatted_metrics, use_container_width=True)

        backtest_csv = backtest_results["metrics"].to_csv().encode('utf-8')
        st.download_button(
            label="📥 Baixar Resultados do Backtest (CSV)",
            data=backtest_csv,
            file_name="metricas_backtest.csv",
            mime="text/csv"
        )
        st.caption("Nota: O backtest utiliza pesos fixos buy-and-hold. Desempenho passado não garante desempenho futuro.")

    # TAB 5: Institucional / Banking
    with tab5:
        st.markdown("### 🏦 Painel Institucional — Ferramentas para Mesa de Risco e Gestão de Ativos")
        st.caption(
            "Este painel reúne análises usadas por mesas de risco, áreas de gestão de carteiras, "
            "compliance regulatório (Basileia III/IV) e advisory de bancos e gestoras."
        )

        # ---- SECTION 1: Risk Contribution ----
        st.markdown("---")
        st.subheader("1. Decomposição de Risco por Ativo (Risk Contribution)")
        st.markdown(
            "Indica **qual ativo está gerando mais risco** para a carteira. "
            "Se um ativo com 10% de alocação gera 50% do risco total, ele é o principal fator de instabilidade. "
            "Fundamental para decisões de rebalanceamento e limites de concentração."
        )

        risk_contrib_df = calculate_risk_contribution(max_sharpe_port["weights"], cov_matrix)
        col_rc_chart, col_rc_table = st.columns([1.3, 1])
        with col_rc_chart:
            st.plotly_chart(plot_risk_contribution(risk_contrib_df), use_container_width=True)
        with col_rc_table:
            st.dataframe(risk_contrib_df, use_container_width=True)

        # ---- SECTION 2: Rolling Window Analytics ----
        st.markdown("---")
        st.subheader("2. Análise de Janela Móvel (Rolling Window)")
        st.markdown(
            "Mostra como a volatilidade e o Sharpe da carteira evoluíram ao longo do tempo. "
            "Fundamental para identificar **mudanças de regime de mercado**, períodos de calma e períodos de estresse."
        )

        rolling_window = st.select_slider(
            "Janela Móvel (dias úteis)",
            options=[21, 42, 63, 126, 252],
            value=63,
            help="21 = ~1 mês, 63 = ~3 meses, 126 = ~6 meses, 252 = ~1 ano"
        )

        port_daily_ret_rolling = calculate_portfolio_daily_returns(daily_returns, max_sharpe_port["weights"].values)
        rolling_df = calculate_rolling_metrics(port_daily_ret_rolling, window=rolling_window, risk_free_rate=risk_free_rate)

        if not rolling_df.empty:
            st.plotly_chart(plot_rolling_metrics(rolling_df), use_container_width=True)
        else:
            st.warning("Dados insuficientes para a janela móvel selecionada.")

        # ---- SECTION 3: Tracking Error & Information Ratio ----
        st.markdown("---")
        st.subheader(f"3. Tracking Error & Information Ratio (vs {benchmark})")
        st.markdown(
            "**Tracking Error** mede o quanto o retorno da carteira se desvia do benchmark (desvio-padrão dos retornos ativos). "
            "**Information Ratio** mede se essa diferença é positiva (gera alfa) por unidade de risco ativo. "
            "Bancos utilizam estes indicadores para avaliar a performance de gestores relativamente ao mandato."
        )

        if bench_returns is not None and not bench_returns.empty:
            te_ir = calculate_tracking_error_and_ir(port_daily_ret_rolling, bench_returns)
            col_te, col_ir = st.columns(2)
            with col_te:
                st.metric(
                    label=f"📏 Tracking Error Anualizado (vs {benchmark})",
                    value=f"{te_ir['tracking_error']*100:.2f}%",
                    help="Desvio-padrão anualizado dos retornos ativos (carteira - benchmark)."
                )
            with col_ir:
                ir_val = te_ir['information_ratio']
                st.metric(
                    label="📐 Information Ratio",
                    value=f"{ir_val:.2f}",
                    delta="Bom" if ir_val > 0.5 else ("Neutro" if ir_val > 0 else "Negativo"),
                    help="Excesso de retorno anualizado / Tracking Error. IR > 0.5 é considerado bom."
                )
        else:
            st.info("Benchmark não disponível para cálculo de Tracking Error.")

        # ---- SECTION 4: Risk Parity vs Markowitz ----
        st.markdown("---")
        st.subheader("4. Comparação Multidimensional: Markowitz vs Risk Parity vs Mínima Vol")
        st.markdown(
            "**Risk Parity (Paridade de Risco)** é uma estratégia de alocação onde cada ativo contribui igualmente para o risco total da carteira. "
            "Popularizada por Ray Dalio (Bridgewater All Weather), é amplamente usada por fundos multimercados e tesourarias de bancos. "
            "Compare visualmente as estratégias no gráfico radar abaixo."
        )

        portfolios_comparison = {
            "Máximo Sharpe": max_sharpe_port,
            "Mínima Volatilidade": gmv_port,
            "Risk Parity": risk_parity_port
        }
        st.plotly_chart(
            plot_multi_portfolio_comparison(portfolios_comparison, daily_returns, risk_free_rate),
            use_container_width=True
        )

        # Risk Parity allocation details
        st.markdown("**Alocação da Carteira Risk Parity:**")
        rp_weights_df = pd.DataFrame({
            "Ativo": risk_parity_port["weights"].index,
            "Peso (%)": (risk_parity_port["weights"].values * 100).round(2)
        }).set_index("Ativo")
        col_rp_pie, col_rp_tbl = st.columns([1, 1])
        with col_rp_pie:
            st.plotly_chart(
                plot_portfolio_donut(risk_parity_port["weights"], title="Alocação Risk Parity"),
                use_container_width=True
            )
        with col_rp_tbl:
            st.dataframe(rp_weights_df, use_container_width=True)
            st.caption(
                f"Sharpe Risk Parity: **{risk_parity_port['sharpe']:.2f}** | "
                f"Volatilidade: **{risk_parity_port['volatility']*100:.2f}%** | "
                f"Retorno: **{risk_parity_port['return']*100:.2f}%**"
            )

        # ---- SECTION 5: Stress Testing ----
        st.markdown("---")
        st.subheader("5. Teste de Estresse Histórico (Stress Testing)")
        st.markdown(
            "Simula como a carteira teria se comportado durante as **piores crises da história recente do mercado brasileiro e global**. "
            "Reguladores exigem que bancos demonstrem a resiliência de suas posições a cenários de estresse (ICAAP, Circular BCB 3.846). "
            "Os cenários são ancorados em dados históricos reais dos períodos indicados."
        )

        stress_results = run_historical_stress_test(
            daily_returns=daily_returns,
            weights=max_sharpe_port["weights"],
            initial_investment=initial_investment
        )

        stress_summary = stress_results["summary"]
        if not stress_summary.empty:
            st.plotly_chart(plot_stress_test_waterfall(stress_summary), use_container_width=True)

            st.markdown("**Detalhamento dos Cenários:**")
            display_cols = ["Cenário", "Período", "Dias Úteis", "Retorno Acumulado (%)", "Max Drawdown (%)", "Pior Dia (%)", "Impacto (R$)", "Descrição"]
            available_cols = [c for c in display_cols if c in stress_summary.columns]
            stress_display = stress_summary[available_cols].copy()

            for col in ["Retorno Acumulado (%)", "Max Drawdown (%)", "Pior Dia (%)"]:
                if col in stress_display.columns:
                    stress_display[col] = stress_display[col].map("{:.2f}%".format)
            if "Impacto (R$)" in stress_display.columns:
                stress_display["Impacto (R$)"] = stress_display["Impacto (R$)"].map("R$ {:,.2f}".format)

            st.dataframe(stress_display, use_container_width=True)
        else:
            st.info(
                "Nenhum dos cenários de estresse pré-definidos se sobrepõe ao período de dados selecionado. "
                "Amplie o período de análise na barra lateral para incluir crises históricas (ex: 2008-2024)."
            )

        # ---- SECTION 6: Custom Shock Simulator ----
        st.markdown("---")
        st.subheader("6. Simulador de Choque Personalizado")
        st.markdown(
            "Aplique choques hipotéticos a cada ativo e veja o impacto instantâneo na carteira. "
            "Útil para cenários *what-if* (ex: \"O que acontece se Petrobras cair 30% e VALE sobe 10%?\")."
        )

        shock_cols = st.columns(min(len(tickers), 4))
        shock_map = {}
        active_tickers = [t for t in tickers if t in max_sharpe_port["weights"].index]
        for i, ticker in enumerate(active_tickers):
            with shock_cols[i % len(shock_cols)]:
                shock_pct = st.number_input(
                    f"Choque {ticker} (%)",
                    min_value=-100.0,
                    max_value=100.0,
                    value=0.0,
                    step=5.0,
                    key=f"shock_{ticker}"
                )
                shock_map[ticker] = shock_pct / 100.0

        if st.button("🧪 Simular Choque", type="primary"):
            shock_result = run_custom_shock(
                annualized_returns, cov_matrix,
                max_sharpe_port["weights"], shock_map, initial_investment
            )
            col_shock_impact, col_shock_detail = st.columns([1, 2])
            with col_shock_impact:
                pct_val = shock_result["portfolio_shock_pct"]
                color = "inverse" if pct_val < 0 else "normal"
                st.metric(
                    label="Impacto na Carteira",
                    value=f"{pct_val:.2f}%",
                    delta=f"R$ {shock_result['portfolio_loss_brl']:,.2f}"
                )
            with col_shock_detail:
                st.dataframe(shock_result["per_asset"], use_container_width=True)

    # TAB 6: Metodologia & MPT
    with tab6:
        render_mpt_theory_explanation()

    st.markdown("---")
    render_regulatory_disclaimer()


if __name__ == "__main__":
    main()
