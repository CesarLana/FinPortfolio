"""
Plotly Charting module for FinPortfolio Streamlit UI.
Provides interactive financial charts with modern dark styling.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Dark theme visual styling tokens
BG_COLOR = "#0E1117"
CARD_BG = "#161B22"
TEXT_COLOR = "#E6EDF3"
GRID_COLOR = "#21262D"


def _set_dark_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    """Applies clean dark theme styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Roboto, sans-serif"),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor=GRID_COLOR
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor=GRID_COLOR
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor=GRID_COLOR,
            borderwidth=1
        )
    )
    return fig


def plot_normalized_prices(prices_normalized: pd.DataFrame) -> go.Figure:
    """Plots normalized price series over time (Base 100)."""
    fig = go.Figure()
    for col in prices_normalized.columns:
        fig.add_trace(go.Scatter(
            x=prices_normalized.index,
            y=prices_normalized[col],
            mode='lines',
            name=col,
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>"
        ))
    return _set_dark_layout(
        fig,
        title="Evolução de Preços Normalizada (Base 100)",
        xaxis_title="Data",
        yaxis_title="Preço Normalizado"
    )


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plots correlation matrix heatmap with annotations."""
    z = corr_matrix.values
    x = corr_matrix.columns.tolist()
    y = corr_matrix.index.tolist()

    z_text = np.round(z, 2).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale='RdBu_r',
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlação")
    ))

    return _set_dark_layout(
        fig,
        title="Matriz de Correlação entre Ativos",
        xaxis_title="Ativos",
        yaxis_title="Ativos"
    )


def plot_efficient_frontier_and_simulations(
    simulations_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    gmv_port: Dict[str, Any],
    max_sharpe_port: Dict[str, Any],
    risk_free_rate: float
) -> go.Figure:
    """Plots Monte Carlo simulated portfolios, Efficient Frontier curve, GMV & Max Sharpe portfolios."""
    fig = go.Figure()

    # 1. Monte Carlo Simulated Portfolios Scatter
    fig.add_trace(go.Scatter(
        x=simulations_df['volatility'] * 100,
        y=simulations_df['return'] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=simulations_df['sharpe'],
            colorscale='Viridis',
            colorbar=dict(title="Índice de Sharpe"),
            showscale=True,
            opacity=0.6
        ),
        name="Carteiras Simuladas",
        hovertemplate=(
            "<b>Carteira Simulada</b><br>" +
            "Volatilidade: %{x:.2f}%<br>" +
            "Retorno: %{y:.2f}%<br>" +
            "Sharpe: %{marker.color:.2f}<extra></extra>"
        )
    ))

    # 2. Efficient Frontier Curve
    if not frontier_df.empty:
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'] * 100,
            y=frontier_df['return'] * 100,
            mode='lines',
            line=dict(color='#00E676', width=3, dash='dash'),
            name="Fronteira Eficiente",
            hovertemplate="<b>Fronteira Eficiente</b><br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
        ))

    # 3. Minimum Volatility Portfolio Marker (Green Star)
    fig.add_trace(go.Scatter(
        x=[gmv_port['volatility'] * 100],
        y=[gmv_port['return'] * 100],
        mode='markers',
        marker=dict(size=16, color='#00E676', symbol='star', line=dict(color='white', width=1.5)),
        name=f"Mínima Volatilidade (Sharpe: {gmv_port['sharpe']:.2f})",
        hovertemplate=(
            "<b>Carteira Mínima Volatilidade</b><br>" +
            f"Volatilidade: {gmv_port['volatility']*100:.2f}%<br>" +
            f"Retorno Esperado: {gmv_port['return']*100:.2f}%<br>" +
            f"Sharpe: {gmv_port['sharpe']:.2f}<extra></extra>"
        )
    ))

    # 4. Maximum Sharpe Ratio Portfolio Marker (Gold Star)
    fig.add_trace(go.Scatter(
        x=[max_sharpe_port['volatility'] * 100],
        y=[max_sharpe_port['return'] * 100],
        mode='markers',
        marker=dict(size=18, color='#FFD700', symbol='star', line=dict(color='white', width=1.5)),
        name=f"Máximo Sharpe (Sharpe: {max_sharpe_port['sharpe']:.2f})",
        hovertemplate=(
            "<b>Carteira Máximo Sharpe</b><br>" +
            f"Volatilidade: {max_sharpe_port['volatility']*100:.2f}%<br>" +
            f"Retorno Esperado: {max_sharpe_port['return']*100:.2f}%<br>" +
            f"Sharpe: {max_sharpe_port['sharpe']:.2f}<extra></extra>"
        )
    ))

    return _set_dark_layout(
        fig,
        title="Fronteira Eficiente de Markowitz & Simulação Monte Carlo",
        xaxis_title="Risco / Volatilidade Anualizada (%)",
        yaxis_title="Retorno Esperado Anualizado (%)"
    )


def plot_portfolio_weights_comparison(weights_gmv: pd.Series, weights_sharpe: pd.Series) -> go.Figure:
    """Grouped bar chart comparing weights between GMV and Max Sharpe portfolios."""
    df_weights = pd.DataFrame({
        'Ativo': weights_gmv.index,
        'Mínima Volatilidade': weights_gmv.values * 100,
        'Máximo Sharpe': weights_sharpe.values * 100
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_weights['Ativo'],
        y=df_weights['Mínima Volatilidade'],
        name='Mínima Volatilidade',
        marker_color='#00E676',
        text=np.round(df_weights['Mínima Volatilidade'], 1).astype(str) + '%',
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=df_weights['Ativo'],
        y=df_weights['Máximo Sharpe'],
        name='Máximo Sharpe',
        marker_color='#FFD700',
        text=np.round(df_weights['Máximo Sharpe'], 1).astype(str) + '%',
        textposition='auto'
    ))

    fig.update_layout(barmode='group')
    return _set_dark_layout(
        fig,
        title="Composição dos Pesos da Carteira (%)",
        xaxis_title="Ativo",
        yaxis_title="Alocação (%)"
    )


def plot_portfolio_donut(weights: pd.Series, title: str = "Alocação da Carteira") -> go.Figure:
    """Plots interactive Donut chart of asset weights."""
    clean_w = weights[weights > 0.001]
    fig = go.Figure(data=[go.Pie(
        labels=clean_w.index,
        values=clean_w.values * 100,
        hole=0.45,
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#0E1117', width=2))
    )])

    return _set_dark_layout(fig, title=title)


def plot_backtest_wealth_curves(wealth_curves: pd.DataFrame) -> go.Figure:
    """Plots comparative wealth evolution curve (R$)."""
    fig = go.Figure()
    colors = ['#FFD700', '#29B6F6', '#AB47BC', '#FF7043']

    for i, col in enumerate(wealth_curves.columns):
        fig.add_trace(go.Scatter(
            x=wealth_curves.index,
            y=wealth_curves[col],
            mode='lines',
            name=col,
            line=dict(width=2.5 if col == "Carteira Otimizada" else 1.8, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{col}</b>: R$ %{{y:,.2f}}<extra></extra>"
        ))

    return _set_dark_layout(
        fig,
        title="Evolução do Patrimônio (Backtest Histórico)",
        xaxis_title="Data",
        yaxis_title="Valor Acumulado (R$)"
    )


def plot_backtest_drawdowns(drawdowns: pd.DataFrame) -> go.Figure:
    """Plots drawdown percentages over time."""
    fig = go.Figure()
    colors = ['#FFD700', '#29B6F6', '#AB47BC', '#FF7043']

    for i, col in enumerate(drawdowns.columns):
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns[col] * 100,
            mode='lines',
            name=col,
            fill='tozeroy' if col == "Carteira Otimizada" else None,
            line=dict(width=1.5, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}%<extra></extra>"
        ))

    return _set_dark_layout(
        fig,
        title="Histórico de Queda em Relação ao Topo (Drawdown %)",
        xaxis_title="Data",
        yaxis_title="Drawdown (%)"
    )


def plot_returns_distribution_with_var(
    daily_returns: pd.Series,
    var_95: float,
    var_99: float,
    cvar_95: float
) -> go.Figure:
    """Plots daily return distribution histogram with VaR and CVaR lines."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_returns * 100,
        nbinsx=50,
        name="Retorno Diário",
        marker_color="#29B6F6",
        opacity=0.75
    ))

    # VaR 95% line
    fig.add_vline(
        x=var_95 * 100,
        line_width=2,
        line_dash="dash",
        line_color="#FF9800",
        annotation_text=f"VaR 95%: {var_95*100:.2f}%",
        annotation_position="top left"
    )

    # VaR 99% line
    fig.add_vline(
        x=var_99 * 100,
        line_width=2,
        line_dash="dash",
        line_color="#F44336",
        annotation_text=f"VaR 99%: {var_99*100:.2f}%",
        annotation_position="bottom left"
    )

    # CVaR 95% line
    fig.add_vline(
        x=cvar_95 * 100,
        line_width=2,
        line_dash="dot",
        line_color="#D50000",
        annotation_text=f"CVaR 95%: {cvar_95*100:.2f}%",
        annotation_position="top right"
    )

    return _set_dark_layout(
        fig,
        title="Distribuição dos Retornos Diários & Value at Risk (VaR / CVaR)",
        xaxis_title="Retorno Diário (%)",
        yaxis_title="Frequência (Dias)"
    )


def plot_risk_contribution(risk_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing % risk contribution per asset."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=risk_df["Ativo"],
        x=risk_df["Contribuição ao Risco (%)"],
        orientation='h',
        marker_color='#FF7043',
        text=risk_df["Contribuição ao Risco (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition='auto'
    ))

    # Add a vertical line at equal contribution target
    n = len(risk_df)
    if n > 0:
        fig.add_vline(
            x=100.0 / n,
            line_dash="dash",
            line_color="#00E676",
            annotation_text=f"Meta Risk Parity ({100/n:.1f}%)"
        )

    return _set_dark_layout(
        fig,
        title="Decomposição de Risco por Ativo (Risk Contribution %)",
        xaxis_title="Contribuição ao Risco Total (%)",
        yaxis_title="Ativo"
    )


def plot_rolling_metrics(rolling_df: pd.DataFrame) -> go.Figure:
    """Plots rolling volatility and rolling Sharpe on dual y-axes."""
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=rolling_df.index,
            y=rolling_df["rolling_vol"] * 100,
            mode='lines',
            name="Volatilidade Móvel (%)",
            line=dict(color="#FF7043", width=2),
            hovertemplate="Vol: %{y:.2f}%<extra></extra>"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_df.index,
            y=rolling_df["rolling_sharpe"],
            mode='lines',
            name="Sharpe Móvel",
            line=dict(color="#29B6F6", width=2),
            hovertemplate="Sharpe: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )

    # Zero line for Sharpe reference
    fig.add_hline(y=0.0, line_dash="dot", line_color="#666", secondary_y=True)

    fig.update_layout(
        title=dict(text="<b>Análise de Janela Móvel (Rolling Window)</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Roboto, sans-serif"),
        legend=dict(bgcolor="rgba(22, 27, 34, 0.8)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(title_text="Data", gridcolor=GRID_COLOR, showline=True, linecolor=GRID_COLOR)
    fig.update_yaxes(title_text="Volatilidade Anualizada (%)", secondary_y=False, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Índice de Sharpe", secondary_y=True, gridcolor=GRID_COLOR)

    return fig


def plot_stress_test_waterfall(stress_df: pd.DataFrame) -> go.Figure:
    """Waterfall chart of stress scenario portfolio impacts."""
    if stress_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Nenhum cenário de stress se sobrepõe ao período de dados selecionado.",
                          showarrow=False, font=dict(size=16, color=TEXT_COLOR))
        return _set_dark_layout(fig, title="Stress Test — Nenhum Cenário Aplicável")

    scenarios = stress_df["Cenário"].tolist()
    impacts = stress_df["Retorno Acumulado (%)"].tolist()

    colors = ['#F44336' if v < 0 else '#4CAF50' for v in impacts]

    fig = go.Figure(go.Bar(
        x=scenarios,
        y=impacts,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in impacts],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_width=1, line_color="#666")

    return _set_dark_layout(
        fig,
        title="Impacto da Carteira em Cenários de Estresse Histórico",
        xaxis_title="Cenário de Crise",
        yaxis_title="Retorno Acumulado (%)"
    )


def plot_multi_portfolio_comparison(
    portfolios: dict,
    daily_returns: pd.DataFrame,
    risk_free_rate: float = 0.1075,
    trading_days: int = 252
) -> go.Figure:
    """Radar chart comparing key metrics across GMV, Max Sharpe, Risk Parity, and 1/N portfolios."""
    import plotly.graph_objects as go

    categories = ["Retorno (%)", "Sharpe", "1 - Vol (%)", "1 - |MDD| (%)", "Diversificação (ENC)"]

    fig = go.Figure()
    colors = ['#FFD700', '#00E676', '#AB47BC', '#29B6F6']

    for idx, (name, port) in enumerate(portfolios.items()):
        w = port['weights']
        n = len(w)
        from src.risk import calculate_diversification_metrics
        enc = calculate_diversification_metrics(w)['effective_n']

        ret_pct = port['return'] * 100
        sharpe = port['sharpe']
        vol_inv = max(0, 100 - port['volatility'] * 100)

        # Estimate MDD from annualized vol (rough approximation for radar)
        mdd_inv = max(0, 100 - abs(port['volatility']) * 100)

        values = [ret_pct, sharpe, vol_inv, mdd_inv, enc / n * 100]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, gridcolor=GRID_COLOR, color=TEXT_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR)
        ),
        title=dict(text="<b>Comparação Multidimensional de Carteiras</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, Roboto, sans-serif"),
        legend=dict(bgcolor="rgba(22, 27, 34, 0.8)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    return fig
