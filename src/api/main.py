import os
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse
)

app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("models")
model = None
scaler = None
metrics = None
model_version = None
TICKER = "AAPL"
SEQ_LENGTH = 60


def load_latest_model():
    global model, scaler, metrics, model_version

    model_files = sorted(MODEL_DIR.glob("lstm_*.keras"), reverse=True)

    if not model_files:
        raise FileNotFoundError("No model files found")

    latest_model_path = model_files[0]
    full_version = latest_model_path.stem.replace("lstm_", "")
    model_version = full_version

    scaler_path = MODEL_DIR / f"scaler_{full_version}.pkl"
    metrics_path = MODEL_DIR / f"metrics_{full_version.replace(f'{TICKER}_', '')}.json"

    print(f"Loading model: {latest_model_path}")
    model = tf.keras.models.load_model(latest_model_path)

    print(f"Loading scaler: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Loading metrics: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"✓ Model loaded successfully: version {model_version}")


def get_recent_data(days=150):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    stock_data = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if len(stock_data) < SEQ_LENGTH:
        raise ValueError(f"Not enough data. Need at least {SEQ_LENGTH} days")

    return stock_data['Close'].values


@app.on_event("startup")
async def startup_event():
    try:
        load_latest_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Stock Price Prediction API",
        "ticker": TICKER,
        "model": "LSTM",
        "version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_version or "unknown"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    if model is None or metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_type=metrics.get("model", "LSTM"),
        ticker=metrics.get("ticker", TICKER),
        version=model_version,
        training_date=metrics.get("timestamp", "unknown"),
        metrics={
            "test_mae": metrics.get("test_mae"),
            "test_rmse": metrics.get("test_rmse"),
            "test_mape": metrics.get("test_mape"),
            "test_r2": metrics.get("test_r2"),
            "training_time_seconds": metrics.get("training_time_seconds")
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        recent_prices = get_recent_data()

        scaled_data = scaler.transform(recent_prices.reshape(-1, 1))

        last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

        predictions = []
        dates = []
        current_sequence = last_sequence.copy()

        last_date = datetime.now()

        for i in range(request.days):
            pred_scaled = model.predict(current_sequence, verbose=0)

            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred_price))

            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date.strftime("%Y-%m-%d"))

            current_sequence = np.append(current_sequence[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

        return PredictionResponse(
            ticker=TICKER,
            prediction_date=datetime.now().strftime("%Y-%m-%d"),
            predictions=predictions,
            dates=dates,
            model_version=f"lstm_{model_version}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/reload")
async def reload_model():
    try:
        load_latest_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
