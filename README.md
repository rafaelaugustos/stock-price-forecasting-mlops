# Stock Price Prediction API

API de previsão de preços de ações utilizando LSTM (Long Short-Term Memory) com estratégia completa de MLOps.

## 🎯 Objetivo

Projeto desenvolvido para a prova substitutiva da Fase 5 - Machine Learning Engineering (FIAP). Implementa um pipeline completo de MLOps para previsão de preços de ações da Apple (AAPL).

## 🚀 API em Produção

**URL:** https://stock-price-forecasting-mlops.onrender.com

### Endpoints Disponíveis

- `GET /` - Informações da API
- `GET /health` - Status de saúde da API
- `GET /model/info` - Métricas do modelo treinado
- `POST /predict` - Realizar previsões
- `GET /docs` - Documentação interativa (Swagger)

### Exemplo de Uso

```bash
curl -X POST https://stock-price-forecasting-mlops.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"days": 5}'
```

## 📊 Modelo

### Arquitetura LSTM

- **Tipo:** Long Short-Term Memory (Rede Neural Recorrente)
- **Framework:** TensorFlow/Keras
- **Entrada:** Sequências de 60 dias de preços históricos
- **Saída:** Previsão do próximo dia

### Métricas de Performance

- **MAE:** $5.67
- **RMSE:** $7.27
- **MAPE:** 2.44%
- **R²:** 0.864

### Estrutura da Rede

```
LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(25) -> Dense(1)
```

## 🏗️ Arquitetura MLOps

### 1. Coleta de Dados
- Fonte: Yahoo Finance API (yfinance)
- Ticker: AAPL
- Período: 5 anos de dados históricos
- Atualização: Dados coletados em tempo real para previsões

### 2. Treinamento do Modelo
- Script: `src/models/train_lstm.py`
- Normalização: MinMaxScaler (0-1)
- Divisão: 90 dias para teste
- Validação: 10% dos dados de treino
- Épocas: 50
- Batch size: 32

### 3. Serialização
- Modelo: `.keras` (formato nativo TensorFlow)
- Scaler: `.pkl` (pickle)
- Métricas: `.json`

### 4. API REST
- Framework: FastAPI
- Validação: Pydantic
- CORS: Habilitado
- Documentação: Swagger automático

### 5. Containerização
- Docker + docker-compose
- Imagem base: Python 3.13-slim
- Otimização: Multi-stage build

### 6. Deploy
- Plataforma: Render
- Auto-deploy: Push to main branch
- Health checks: Automáticos
- Logs: Centralizados

## 🛠️ Tecnologias

- **Python 3.12+**
- **TensorFlow** - Deep Learning
- **FastAPI** - API REST
- **Pydantic** - Validação de dados
- **yfinance** - Coleta de dados financeiros
- **scikit-learn** - Pré-processamento
- **Docker** - Containerização
- **Render** - Cloud deployment

## 📁 Estrutura do Projeto

```
├── src/
│   ├── api/
│   │   ├── main.py          # API FastAPI
│   │   └── schemas.py       # Schemas Pydantic
│   ├── data/
│   │   └── data_collector.py
│   └── models/
│       └── train_lstm.py    # Script de treinamento
├── models/                   # Modelos treinados
│   ├── lstm_AAPL_*.keras
│   ├── scaler_AAPL_*.pkl
│   └── metrics_*.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 Instalação Local

### Pré-requisitos
- Python 3.12+
- pip

### Passo a Passo

1. Clone o repositório:
```bash
git clone https://github.com/rafaelaugustos/stock-price-forecasting-mlops.git
cd stock-price-forecasting-mlops
```

2. Crie ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale dependências:
```bash
pip install -r requirements.txt
```

4. Treine o modelo (opcional):
```bash
python -m src.models.train_lstm
```

5. Inicie a API:
```bash
uvicorn src.api.main:app --reload
```

A API estará disponível em: http://localhost:8000

## 🐳 Docker

### Build e Run

```bash
docker-compose up --build
```

### Apenas Run

```bash
docker-compose up
```

A API estará disponível em: http://localhost:8000

## 📈 Workflow MLOps

```
1. Coleta de Dados (Yahoo Finance)
        ↓
2. Pré-processamento (MinMaxScaler)
        ↓
3. Treinamento LSTM (TensorFlow)
        ↓
4. Avaliação e Métricas
        ↓
5. Serialização (Model + Scaler)
        ↓
6. API REST (FastAPI)
        ↓
7. Containerização (Docker)
        ↓
8. Deploy Cloud (Render)
        ↓
9. Monitoramento (Logs + Health Checks)
```

## 🔄 CI/CD

- **Trigger:** Push para branch main
- **Build:** Automático via Render
- **Deploy:** Automático após build bem-sucedido
- **Health Check:** Verificação contínua da API

## 📊 Monitoramento

- Health check endpoint: `/health`
- Logs centralizados no Render
- Métricas do modelo disponíveis via `/model/info`

## 🎓 Autor

Projeto desenvolvido para FIAP - Fase 5 - Machine Learning Engineering

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais.
