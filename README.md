# 📈 FinPortfolio — Análise e Otimização de Carteiras de Investimentos

[![Tests](https://github.com/CesarLana/FinPortfolio/actions/workflows/test.yml/badge.svg)](https://github.com/CesarLana/FinPortfolio/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Aplicação interativa em Python voltada para análise estatística de ativos e otimização de portfólios baseada na **Teoria Moderna de Carteiras (Markowitz MPT)**, métricas de risco e visualização de dados via dashboard em **Streamlit**.

---

## 🎯 Funcionalidades

1. **Otimização de Carteiras**:
   - **Mínima Volatilidade (GMV):** Portfólio com a menor variância histórica calculada.
   - **Máximo Índice de Sharpe:** Portfólio tangencial que maximiza a relação retorno esperado / risco.
   - **Risk Parity (Paridade de Risco):** Alocação buscando contribuição igualitária de risco por ativo via otimizador SLSQP (Scipy).
   - **Fronteira Eficiente:** Curva teórica de risco-retorno com restrições de pesos configuráveis.

2. **Gestão de Risco & Análise Estatística**:
   - **Métricas de Cauda:** Value at Risk (VaR Histórico e Paramétrico a 95% e 99%) e Conditional VaR (CVaR / Expected Shortfall).
   - **Métricas de Performance:** Volatilidade anualizada, Sharpe Ratio, Tracking Error, Beta e Alpha de Jensen.
   - **Decomposição de Risco:** Marginal Risk Contribution e Component Risk por ativo.
   - **Cenários Históricos:** Avaliação do impacto da carteira em períodos de estresse passados (ex: Crise de 2008, COVID-19).

3. **Coleta de Dados**:
   - Obtenção de séries temporais de preços de fechamento ajustados via biblioteca `yfinance` para ações da B3 (`.SA`) e índices globais.
   - Tratamento de retornos logarítmicos, alinhamento de datas e tratamento de feriados.

4. **Simulação de Carteiras**:
   - Geração de combinações de pesos aleatórios para mapeamento visual da dispersão risco-retorno e contraste com a Fronteira Eficiente.

5. **Interface Gráfica (Streamlit + Plotly)**:
   - Dashboard responsivo com gráficos dinâmicos de fronteira de Sharpe, correlação e distribuição de pesos.

---

## ⚠️ Premissas e Limitações do Modelo

- **Sensibilidade a Erros de Estimação:** O modelo clássico de Markowitz utiliza dados históricos passados para estimar retornos esperados e matriz de covariância. Retornos passados não garantem retornos futuros e a matriz de covariância pode sofrer instabilidade em regimes de mercado atípicos.
- **Distribuição de Retornos:** Os cálculos paramétricos assumem distribuição aproximadamente normal dos retornos, embora caudas pesadas (fat tails) sejam comuns no mercado financeiro. Por isso, métricas não-paramétricas como o VaR Histórico e CVaR também são disponibilizadas.
- **Fonte de Dados:** Os dados de cotações são extraídos do Yahoo Finance via biblioteca `yfinance`, uma fonte pública de fins educacionais/pessoais sujeita a delay e inconsistências de cotações intradiárias.

---

## 💻 Como Executar Localmente

### Opção 1: Via Docker (Recomendado)

```bash
git clone https://github.com/CesarLana/FinPortfolio.git
cd FinPortfolio

docker compose up --build -d
```
Acesse no navegador: `http://localhost:8501`

### Opção 2: Com Ambiente Virtual Python

```bash
git clone https://github.com/CesarLana/FinPortfolio.git
cd FinPortfolio

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 🧪 Testes Automatizados

O projeto conta com suíte de testes unitários validando as rotinas matemáticas de otimização, cálculo de covariância e métricas de risco:

```bash
pytest tests/ -v
```

Uma esteira de **CI (Continuous Integration) via GitHub Actions** executa os testes automaticamente a cada commit ou pull request no branch `main`.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE) para mais informações.
