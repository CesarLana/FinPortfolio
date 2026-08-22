"""
Metrics and Educational documentation component for Streamlit dashboard.
Provides KPI cards, disclaimers, and mathematical explanations of MPT.
"""

import streamlit as st
from typing import Dict, Any


def render_metric_cards(metrics: Dict[str, float]) -> None:
    """Renders top level KPI metrics in a sleek 2-row layout."""
    # Row 1: Core Return & Risk Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Retorno Esperado (CAGR)",
            value=f"{metrics['expected_return']*100:.2f}%",
            help="Taxa composta anualizada de retorno esperado."
        )

    with col2:
        st.metric(
            label="📉 Volatilidade Anualizada",
            value=f"{metrics['volatility']*100:.2f}%",
            help="Desvio padrão anualizado dos retornos diários."
        )

    with col3:
        st.metric(
            label="⚡ Índice de Sharpe",
            value=f"{metrics['sharpe_ratio']:.2f}",
            help="Excesso de retorno por unidade de volatilidade total acima da Taxa Selic."
        )

    with col4:
        st.metric(
            label="🎯 Índice de Sortino",
            value=f"{metrics['sortino_ratio']:.2f}",
            help="Excesso de retorno penalizando apenas a volatilidade negativa (Downside Risk)."
        )

    # Row 2: Advanced Risk & Diversification
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="⚠️ Max Drawdown",
            value=f"{metrics['max_drawdown']*100:.2f}%",
            help="Maior perda acumulada do pico até o fundo histórico."
        )

    with col6:
        st.metric(
            label="🛑 VaR Histórico (95%)",
            value=f"{metrics['var_95']*100:.2f}%",
            help="Perda máxima diária esperada com 95% de confiança estatística."
        )

    with col7:
        st.metric(
            label="🚨 CVaR / Expected Shortfall (95%)",
            value=f"{metrics['cvar_95']*100:.2f}%",
            help="Média das perdas nos 5% piores dias do histórico."
        )

    with col8:
        st.metric(
            label="🌐 Ativos Efetivos (ENC)",
            value=f"{metrics['effective_n']:.1f}",
            help="Número efetivo de ativos na carteira (1 / HHI), indicando nível de diversificação."
        )


def render_regulatory_disclaimer() -> None:
    """Renders mandatory financial and regulatory educational notice."""
    st.warning(
        "⚠️ **Aviso de Isenção de Responsabilidade (Disclaimer Legal & Educacional)**\n\n"
        "Este projeto possui finalidade estritamente **educacional e de análise quantitativa**. "
        "Os resultados, otimizações e simulações apresentadas são baseados em dados históricos e "
        "**não constituem recomendação de investimento**, indicação de compra/venda de ativos ou garantia de rentabilidade futura. "
        "Investimentos no mercado financeiro envolvem riscos e podem resultar em perda de capital."
    )


def render_mpt_theory_explanation() -> None:
    """Renders detailed mathematical explanation of Markowitz MPT and its practical limitations."""
    st.markdown("### 📚 Teoria Moderna de Portfólios (MPT) de Harry Markowitz")
    st.markdown(
        "Introduzida por Harry Markowitz em 1952 (vencedor do Prêmio Nobel de Economia em 1990), "
        "a Teoria Moderna de Portfólios demonstra matematicamente que o risco de um ativo individual não deve "
        "ser avaliado isoladamente, mas sim pelo seu **impacto no risco global da carteira**, ponderado pelas "
        "**correlações** entre os ativos."
    )

    st.markdown("#### 📐 Formulários Matemáticos Principais")

    st.latex(r"E[R_p] = \sum_{i=1}^{N} w_i \cdot E[R_i] = \mathbf{w}^T \mathbf{\mu}")
    st.caption("1. Retorno Esperado da Carteira: Média ponderada dos retornos individuais dos ativos.")

    st.latex(r"\sigma_p^2 = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_{ij} = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}")
    st.caption("2. Variância da Carteira: Considera a covariância σ_ij entre todos os pares de ativos.")

    st.latex(r"\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}")
    st.caption("3. Índice de Sharpe: Mede o retorno excedente por unidade de risco assumida acima da taxa livre de risco (R_f).")

    st.latex(r"\text{Sortino Ratio} = \frac{E[R_p] - R_f}{\sigma_{\text{downside}}}")
    st.caption("4. Índice de Sortino: Mede o retorno excedente considerando apenas desvios abaixo do benchmark (risco de queda).")

    st.latex(r"\text{HHI} = \sum_{i=1}^{N} w_i^2, \quad \text{ENC} = \frac{1}{\text{HHI}}")
    st.caption("5. Índice Herfindahl-Hirschman (HHI) e Número Efetivo de Constituintes (ENC): Medem concentração da carteira.")

    st.markdown("---")
    st.markdown("#### ⚠️ Limitações do Modelo de Markowitz no Mundo Real")
    st.info(
        "1. **Dependência de Dados Históricos**: O modelo assume que o comportamento futuro (retornos e covariâncias) "
        "seguirá o padrão histórico passado, o que raramente se sustenta em períodos de crise ou mudanças macroeconômicas.\n"
        "2. **Sensibilidade a Erros de Estimativa**: Pequenas variações na estimativa de retorno esperado podem gerar "
        "mudanças drásticas e instáveis nos pesos otimizados (problema conhecido como *'estimation error maximization'*).\n"
        "3. **Distribuição Normal dos Retornos**: O modelo pressupõe que os retornos seguem uma distribuição normal, "
        "ignorando caudas longas (*fat tails*) e eventos de choques de mercado (*black swans*).\n"
        "4. **Covariância Estática**: A matriz de covariância varia ao longo do tempo. Em cenários de pânico no mercado, "
        "a correlação entre a maioria dos ativos tende a subir rapidamente para +1, reduzindo o benefício da diversificação.\n"
        "5. **Ausência de Custos de Transação e Liquidez**: A otimização pura não considera impostos, taxas de corretagem, "
        "slippage ou iliquidez dos ativos ao rebalancear a carteira."
    )
