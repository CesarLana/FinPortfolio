# 📈 FinPortfolio — Análise e Otimização de Carteiras de Investimentos (Markowitz MPT)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

O **FinPortfolio** é uma plataforma quantitativa profissional de otimização de investimentos. O sistema combina a matemática clássica da **Teoria Moderna de Portfólios (MPT)** com análises de risco institucionais (usadas por tesourarias e assets), coleta automatizada de dados em tempo real e uma interface gráfica interativa.

---

## 🚀 Funcionalidades Principais

1. **Otimização Avançada de Portfólio**:
   - **Mínima Volatilidade (GMV):** Portfólio com o menor risco histórico possível.
   - **Máximo Índice de Sharpe:** Portfólio tangencial que maximiza a relação risco/retorno esperado.
   - **Risk Parity (Paridade de Risco):** Estratégia de alocação onde cada ativo contribui igualmente para o risco total da carteira.
   - **Fronteira Eficiente** com limites mínimos e máximos configuráveis por ativo.

2. **Módulo Institucional & Risk Management**:
   - **Stress Testing Histórico:** Simulação do impacto na carteira em crises reais (ex: 2008, COVID-19, Joesley Day).
   - **Decomposição de Risco (Risk Contribution):** Marginal Risk e Component Risk por ativo.
   - **Métricas de Cauda:** VaR 95/99% (Value at Risk) e CVaR (Expected Shortfall).
   - **Performance Analítica:** Rolling Volatility, Rolling Sharpe, Tracking Error, Information Ratio, Beta e Alpha de Jensen.
   - **Simulador de Choques:** Testes de cenário *What-If* customizados.

3. **Coleta de Dados e Integração**:
   - Integração com a API do Yahoo Finance (`yfinance`) para B3 (`.SA`) e ativos globais.
   - Tratamento automático de fuso horário, dados nulos e feriados de mercado.

4. **Backtest e Simulação Estocástica**:
   - Geração de milhares de portfólios randômicos via Simulação de Monte Carlo.
   - Backtest histórico buy-and-hold comparado a benchmarks (Ibovespa, S&P 500).

5. **Dashboard Interativo (Streamlit + Plotly)**:
   - UI moderna e responsiva (Dark Theme).
   - Gráficos avançados: Fronteira eficiente interativa, Heatmap de Correlação, Donut charts, Waterfall de Stress Test e Radar de Alocação Multidimensional.
   - Exportação de dados (pesos, matrizes, métricas) em formato CSV.

---

## 💻 Como Rodar Localmente (Docker)

O jeito mais fácil de rodar a aplicação sem se preocupar com dependências e versões de bibliotecas é utilizando o Docker e o `docker-compose`.

```bash
# 1. Clone o repositório
git clone https://github.com/CesarLana/FinPortfolio.git
cd FinPortfolio

# 2. Suba o container (Isso instalará as dependências e iniciará o Streamlit)
docker-compose up --build -d

# 3. Acesse a aplicação no seu navegador:
# http://localhost:8501
```

## 🔧 Como Rodar Localmente (Python Nativo)

Caso não queira usar Docker, você pode rodar a aplicação em um ambiente virtual local:

```bash
# 1. Clone o repositório
git clone https://github.com/CesarLana/FinPortfolio.git
cd FinPortfolio

# 2. Crie e ative um ambiente virtual (Opcional, mas recomendado)
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Mac/Linux:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode a aplicação Streamlit
streamlit run app/dashboard.py
```

---

## 🧪 Testes Automatizados

O sistema conta com uma extensa suíte de testes unitários para garantir a integridade dos cálculos financeiros de risco e otimização.

```bash
# Para rodar todos os testes com Pytest:
python -m pytest tests/ -v
```

---

## 📐 Fundamentação Matemática e Modelagem

Este sistema foi construído embasado em modelagem financeira acadêmica estrita:

1. **Retorno do Portfólio ($E[R_p]$)**
   $E[R_p] = \sum_{i=1}^{n} w_i E[R_i]$

2. **Volatilidade do Portfólio ($\sigma_p$)**
   $\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$
   Sendo $\mathbf{\Sigma}$ a matriz de covariância anualizada e $\mathbf{w}$ o vetor de pesos.

3. **Índice de Sharpe**
   $Sharpe = \frac{E[R_p] - R_f}{\sigma_p}$

4. **Risk Parity (Igualdade de Contribuição de Risco)**
   A alocação busca igualar a Contribuição de Risco (CRC) de todos os $N$ ativos. O algoritmo SLSQP minimiza a seguinte função objetivo:
   $\min \sum_{i=1}^{n} \left( \frac{w_i (\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p} - \frac{\sigma_p}{n} \right)^2$

---

> Desenvolvido com Python e IA por César Lana. Código aberto (MIT License).
