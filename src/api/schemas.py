from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    days: int = Field(default=1, ge=1, le=30, description="Number of days to predict (1-30)")


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    predictions: List[float]
    dates: List[str]
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "prediction_date": "2025-10-27",
                "predictions": [230.45, 231.20, 229.80],
                "dates": ["2025-10-28", "2025-10-29", "2025-10-30"],
                "model_version": "lstm_20251027_201201"
            }
        }


class ModelInfoResponse(BaseModel):
    model_type: str
    ticker: str
    version: str
    training_date: str
    metrics: dict

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LSTM",
                "ticker": "AAPL",
                "version": "20251027_201201",
                "training_date": "2025-10-27 20:12:01",
                "metrics": {
                    "test_mae": 5.67,
                    "test_rmse": 7.27,
                    "test_mape": 2.44,
                    "test_r2": 0.864
                }
            }
        }


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str
