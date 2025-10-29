import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import json
import os


def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm_model(ticker='AAPL', seq_length=60, epochs=50, batch_size=32):
    print(f"="*70)
    print(f"TREINAMENTO DE MODELO LSTM - {ticker}")
    print(f"="*70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    print(f"\n1. Coletando dados de {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    print(f"   ✓ {len(stock_data)} registros coletados")

    prices = stock_data['Close'].values.reshape(-1, 1)

    print(f"\n2. Normalizando dados...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    print(f"   ✓ Dados normalizados (escala 0-1)")

    print(f"\n3. Criando sequências...")
    X, y = create_sequences(scaled_data, seq_length)
    print(f"   ✓ {len(X)} sequências criadas (tamanho={seq_length})")

    test_size = 90
    train_size = len(X) - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"\n4. Divisão dos dados:")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste: {len(X_test)} amostras")

    print(f"\n5. Construindo modelo LSTM...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"   ✓ Modelo criado com {model.count_params():,} parâmetros")

    print(f"\n6. Treinando modelo...")
    print(f"   Épocas: {epochs}, Batch size: {batch_size}")

    start_time = datetime.now()

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\n   ✓ Modelo treinado em {training_time:.2f} segundos")

    print(f"\n7. Avaliando modelo...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))

    test_mae = mean_absolute_error(y_test_actual, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    test_mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    test_r2 = r2_score(y_test_actual, test_predictions)

    print(f"\n{'='*70}")
    print(f"MÉTRICAS DE PERFORMANCE")
    print(f"{'='*70}")
    print(f"\nTREINO:")
    print(f"  MAE:  ${train_mae:.2f}")
    print(f"  RMSE: ${train_rmse:.2f}")
    print(f"\nTESTE:")
    print(f"  MAE:  ${test_mae:.2f}")
    print(f"  RMSE: ${test_rmse:.2f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print(f"  R²:   {test_r2:.4f}")
    print(f"{'='*70}")

    mean_price = y_test_actual.mean()
    print(f"\nPreço médio no teste: ${mean_price:.2f}")
    print(f"Erro representa {(test_mae/mean_price)*100:.2f}% do preço médio")

    print(f"\n8. Salvando modelo...")
    os.makedirs('models', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/lstm_{ticker}_{timestamp}.keras'
    scaler_path = f'models/scaler_{ticker}_{timestamp}.pkl'

    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"   ✓ Modelo salvo: {model_path}")
    print(f"   ✓ Scaler salvo: {scaler_path}")

    metrics = {
        'model': 'LSTM',
        'ticker': ticker,
        'timestamp': timestamp,
        'training_time_seconds': training_time,
        'seq_length': seq_length,
        'epochs': epochs,
        'batch_size': batch_size,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'test_r2': float(test_r2),
        'model_file': model_path,
        'scaler_file': scaler_path
    }

    metrics_path = f'models/metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"   ✓ Métricas salvas: {metrics_path}")

    print(f"\n{'='*70}")
    print(f"TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*70}\n")

    return model, scaler, metrics, history


if __name__ == "__main__":
    model, scaler, metrics, history = train_lstm_model(
        ticker='AAPL',
        seq_length=60,
        epochs=50,
        batch_size=32
    )
