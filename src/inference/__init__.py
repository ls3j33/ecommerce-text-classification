"""Inference сервис для классификации товаров."""

from src.inference.main import app, main
from src.inference.model_loader import ModelManager
from src.inference.models import (
    PredictRequest,
    PredictResponse,
    ChangeModelRequest,
    ChangeModelResponse,
    HealthResponse,
)

__all__ = [
    "app",
    "main",
    "ModelManager",
    "PredictRequest",
    "PredictResponse",
    "ChangeModelRequest",
    "ChangeModelResponse",
    "HealthResponse",
]
