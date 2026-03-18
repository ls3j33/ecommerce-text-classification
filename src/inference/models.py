"""Pydantic модели для Inference сервиса."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class PredictRequest(BaseModel):
    """Запрос на предсказание."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Текстовое описание товара",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "New product description text..."
            }
        }
    )


class PredictResponse(BaseModel):
    """Ответ на предсказание."""

    category: str = Field(..., description="Предсказанная категория")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность модели")
    model_version: str = Field(..., description="Версия модели")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "Electronics",
                "confidence": 0.94,
                "model_version": "v1",
            }
        }
    )


class ChangeModelRequest(BaseModel):
    """Запрос на смену модели."""

    version: str = Field(
        ...,
        pattern=r"^v\d+$",
        description="Версия модели для загрузки",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": "v1"
            }
        }
    )


class ChangeModelResponse(BaseModel):
    """Ответ на смену модели."""

    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение")
    model_version: str = Field(..., description="Новая версия модели")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Модель успешно загружена",
                "model_version": "v2",
            }
        }
    )


class HealthResponse(BaseModel):
    """Ответ health check endpoint."""

    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    model_version: Optional[str] = Field(
        None, description="Версия текущей модели"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v1",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Модель ошибки."""

    detail: str = Field(..., description="Описание ошибки")
