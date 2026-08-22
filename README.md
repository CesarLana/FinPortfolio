# 📈 FinPortfolio — Análise e Otimização de Carteiras de Investimentos (Markowitz MPT)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

O **FinPortfolio** é uma aplicação completa de inteligência quantitativa voltada ao mercado financeiro. O sistema realiza coleta de dados históricos de ativos, calcula métricas estatísticas de risco e retorno, executa otimização quadrática baseada na **Teoria Moderna de Portfólios (MPT) de Harry Markowitz**, simula milhares de alocações via **Monte Carlo**, calcula métricas avançadas de risco (**Value at Risk (VaR)** e **Maximum Drawdown**) e realiza **backtests históricos** comparando com benchmarks do mercado como o `BOVA11`.

---

## 🚀 Funcionalidades Principais

1. **Coleta e Tratamento de Dados de Mercado (`src/data.py`)**:
   - Integração com a API do Yahoo Finance (`yfinance`) para ativos da B3 (`.SA`), BDRs, ETFs e ações americanas.
   - Tratamento automatizado de dados ausentes (forward/backward fill), duplicidades no calendário e validação de quantidade mínima de observações sem geração de dados sintéticos.

2. **Cálculo de Retornos (`src/returns.py`)**:
   - Retornos diários simples (aritméticos) e logarítmicos.
   - Retorno acumulado histórico e retorno esperado anualizado composto ($CAGR$).

3. **Métricas de Risco & Matrizes (`src/risk.py`)**:
   - Volatilidade diária e anualizada ($\sigma_a = \sigma_d \sqrt{252}$).
   - Matriz de Covariância ($\mathbf{\Sigma}$) e Matriz de Correlação ($\mathbf{R}$).
   - **Value at Risk (VaR Histórico)** nos níveis de confiança de 95% e 99%.
   - **Maximum Drawdown (MDD)** e identificação dos melhores/piores dias de negociação.

4. **Otimizador de Markowitz (`src/optimization.py`)**:
   - Resolução de Otimização Quadrática via algoritmo SLSQP do `scipy.optimize`.
   - **Carteira de Mínima Volatilidade (GMV)**.
   - **Carteira de Máximo Índice de Sharpe (Tangência)**.
   - Geração gráfica da **Fronteira Eficiente** com suporte a limites mínimos e máximos de exposição por ativo ($w_{\min} \le w_i \le w_{\max}$).

5. **Simulação Monte Carlo (`src/portfolio.py`)**:
   - Geração de até 20.000 carteiras aleatórias respeitando as restrições orçamentárias ($\sum w_i = 100\%$).

6. **Engine de Backtest Histórico (`src/backtest.py`)**:
   - Comparação da performance acumulada da **Carteira Otimizada** vs **Carteira 1/N (Pesos Iguais)** vs **Benchmark (`BOVA11`)**.
   - Curvas de evolução patrimonial (R$), série temporal de Drawdowns e tabela comparativa de Sharpe e CAGR.

7. **Dashboard Interativo em Streamlit (`app/dashboard.py`)**:
   - Interface com tema escuro moderno, gráficos dinâmicos em Plotly, cartões KPI e exportação de resultados para CSV.

---

## 📂 Arquitetura do Projeto

```
finportfolio/
│
├── app/
│   ├── dashboard.py                # Ponto de entrada do Dashboard Streamlit
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py              # Controles laterais (tickers, datas, taxa livre de risco)
│       ├── charts.py               # Gerador de gráficos interativos Plotly
│       └── metrics.py              # Componentes de cartões KPI e explicações teóricas
│
├── src/
│   ├── __init__.py
│   ├── data.py                     # Pipeline de coleta e limpeza de dados
│   ├── returns.py                  # Motores de cálculo de retornos
│   ├── risk.py                     # Módulo estatístico de risco e volatilidade
│   ├── optimization.py             # Algoritmos de otimização quadrática de Markowitz
│   ├── portfolio.py                # Simulação Monte Carlo e resumos de carteira
│   └── backtest.py                 # Simulação e comparação de performance histórica
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py                # Testes unitários do pipeline de dados
│   ├── test_returns.py             # Testes unitários dos retornos
│   ├── test_risk.py                # Testes unitários das métricas de risco
│   ├── test_optimization.py        # Testes das restrições e otimizador de Markowitz
│   └── test_backtest.py            # Testes do motor de backtest
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Notebook demonstrativo passo a passo da matemática MPT
│
├── requirements.txt                # Dependências do projeto
├── Dockerfile                      # Instruções de conteinerização Docker
├── docker-compose.yml              # Orquestração do container
├── .gitignore
├── .dockerignore
└── README.md                       # Documentação completa do projeto
```

---

## 📐 Fundamentação Matemática (Teoria Moderna de Portfólios)

### 1. Retorno Esperado do Portfólio
$$E[R_p] = \mathbf{w}^T \mathbf{\mu} = \sum_{i=1}^{N} w_i E[R_i]$$

### 2. Risco / Volatilidade do Portfólio
$$\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}} = \sqrt{\sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_{ij}}$$

### 3. Índice de Sharpe
$$\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}$$
Onde $R_f$ é a taxa livre de risco anualizada (ex: Selic 10.75%).

### 4. Problema de Otimização da Carteira Tangente (Máximo Sharpe)
$$\max_{\mathbf{w}} \quad \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}$$
$$\text{sujeito a:} \quad \sum_{i=1}^{N} w_i = 1 \quad \text{e} \quad w_{\min} \le w_i \le w_{\max}$$

---

## 🛠️ Instalação e Execução Local

### Pré-requisitos
- Python 3.10+
- `pip` e `git`

### Passo a Passo

1. **Clonar o repositório**:
   ```bash
   git clone https://github.com/SeuUsuario/finportfolio.git
   cd finportfolio
   ```

2. **Criar e ativar um ambiente virtual**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar a suíte de testes unitários**:
   ```bash
   python -m pytest tests/
   ```

5. **Iniciar o Dashboard Streamlit**:
   ```bash
   streamlit run app/dashboard.py
   ```
   A aplicação será aberta automaticamente no navegador em `http://localhost:8501`.

---

## 🐳 Execução via Docker & Docker Compose

Se desejar executar a aplicação conteinerizada:

### Usando Docker Compose:
```bash
docker-compose up --build
```
Acesse `http://localhost:8501` no seu navegador.

### Usando Docker CLI:
```bash
docker build -t finportfolio .
docker run -p 8501:8501 finportfolio
```

---

## 💼 Como Destacar Este Projeto no GitHub & LinkedIn (Dicas de Portfólio)

### No GitHub:
1. **Vídeo de Demonstração / GIF**: Grave um curto vídeo do Dashboard funcionando e converta para GIF no topo do `README.md`.
2. **Commits Estruturados**: Mantenha commits semânticos (ex: `feat: add VaR calculation`, `fix: handle missing yfinance data`).
3. **Tags & Tópicos**: Adicione as seguintes tags no repositório: `finance`, `python`, `markowitz`, `portfolio-optimization`, `quantitative-finance`, `streamlit`, `aws`, `docker`.

### Sugestão de Postagem no LinkedIn:
> 🚀 **Novo Projeto Quant: Análise e Otimização de Carteiras de Investimentos com Python & Markowitz MPT!**
>
> Desenvolvi uma plataforma completa em Python para análise estatística e otimização de carteiras de ativos utilizando a **Teoria Moderna de Portfólios de Markowitz (MPT)**.
>
> 🔹 **Principais Destaques:**
> • Coleta e limpeza de dados de mercado da B3 e ativos globais via API do Yahoo Finance.
> • Otimização quadrática (SciPy SLSQP) para Carteira de Mínima Volatilidade e Máximo Sharpe.
> • Simulação de Monte Carlo com até 20.000 carteiras e plot da Fronteira Eficiente.
> • Análise de risco avançada: Value at Risk (VaR Histórico 95%/99%) e Maximum Drawdown.
> • Engine de Backtest Histórico comparando com Ibovespa (`BOVA11`) e carteira 1/N.
> • Interface web moderna interativa criada com **Streamlit** e **Plotly**.
> • Arquitetura conteinerizada com **Docker** e pronta para deploy na **AWS**.
>
> 🔗 Confira o código-fonte e documentação no GitHub: [Link do Repositório]

---

## ☁️ Guia de Implantação na Nuvem (Deploy AWS)

Este projeto foi projetado modularmente para fácil deploy em nuvem. A seguir estão os passos recomendados para implantação na **AWS**:

### Opção 1: AWS App Runner (Recomendada pela Simplicidade)
O **AWS App Runner** é o serviço gerenciado ideal para contêineres de aplicações web como o Streamlit:
1. Faça push da imagem Docker para o **AWS ECR (Elastic Container Registry)**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker tag finportfolio:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/finportfolio:latest
   docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/finportfolio:latest
   ```
2. No console do **AWS App Runner**, crie um novo serviço apontando para o repositório ECR na porta `8501`.
3. O App Runner fornecerá automaticamente um domínio HTTPS com SSL gerenciado e auto-scaling.

### Opção 2: AWS ECS Fargate + Application Load Balancer (Escala Empresarial)
1. Crie um **Cluster ECS** com o tipo de inicialização **AWS Fargate** (Serverless).
2. Defina uma **Task Definition** com 0.5 vCPU e 1 GB de RAM.
3. Configure um **Application Load Balancer (ALB)** para direcionar o tráfego HTTP/HTTPS na porta 8501.

### CI/CD com GitHub Actions (`.github/workflows/deploy.yml`)
Você pode automatizar os testes e o deploy adicionando um workflow que executa `pytest` a cada `git push` e faz o deploy automático no AWS ECR/App Runner.

---

## ⚠️ Isenção de Responsabilidade (Legal & Regulatory Disclaimer)

> **Este projeto possui finalidade estritamente educacional e de análise quantitativa.**
> Os resultados obtidos são baseados em dados históricos passados e não constituem recomendação de investimento, indicação de compra ou venda de valores mobiliários, ou garantia de rentabilidade futura. Investimentos no mercado financeiro estão sujeitos a riscos de perda de capital.
