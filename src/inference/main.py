"""FastAPI приложение для Inference сервиса."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.config import settings
from src.inference.model_loader import ModelManager
from src.inference.models import (
    ChangeModelRequest,
    ChangeModelResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.logging_config import setup_logger, log

# Глобальный менеджер моделей
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global model_manager

    # Инициализация при запуске
    setup_logger(level="INFO")
    log.info("Запуск Inference сервиса...")

    model_manager = ModelManager()

    # Загружаем текущую модель при старте
    if not model_manager.load_model():
        log.warning("Не удалось загрузить модель при старте")
    else:
        log.info(f"Модель загружена: {model_manager.model_version}")

    yield

    # Очистка при остановке
    log.info("Остановка Inference сервиса...")
    if model_manager:
        model_manager.unload_model()


app = FastAPI(
    title="E-commerce Classification API",
    description="API для классификации товаров электронной коммерции",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Глобальный обработчик исключений."""
    log.error(f"Необработанное исключение: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Внутренняя ошибка сервера"},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Проверка работоспособности сервиса",
)
async def health_check() -> HealthResponse:
    """
    Health Check Endpoint.

    Возвращает статус сервиса и информацию о загруженной модели.
    """
    if model_manager is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None,
        )

    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.model_version,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    },
    tags=["Prediction"],
    summary="Предсказание категории",
    description="Предсказание категории товара по описанию",
)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Предсказание категории товара.

    Принимает текстовое описание товара и возвращает предсказанную категорию
    с уровнем уверенности модели.
    """
    if model_manager is None or not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена",
        )

    try:
        category, confidence = model_manager.predict(request.description)

        return PredictResponse(
            category=category,
            confidence=round(confidence, 4),
            model_version=model_manager.model_version or "unknown",
        )

    except RuntimeError as e:
        log.error(f"Ошибка предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        log.error(f"Неожиданная ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при выполнении предсказания",
        )


@app.post(
    "/change-model",
    response_model=ChangeModelResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Неверная версия"},
        404: {"model": ErrorResponse, "description": "Модель не найдена"},
        500: {"model": ErrorResponse, "description": "Ошибка загрузки"},
    },
    tags=["Model Management"],
    summary="Смена модели",
    description="Загрузка новой версии модели из реестра",
)
async def change_model(request: ChangeModelRequest) -> ChangeModelResponse:
    """
    Смена активной модели.

    Загружает указанную версию модели из реестра.
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис не инициализирован",
        )

    log.info(f"Запрос на смену модели: {request.version}")

    # Проверяем наличие версии в реестре
    available_versions = model_manager.get_available_versions()
    if request.version not in available_versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель версии {request.version} не найдена. "
            f"Доступные версии: {available_versions}",
        )

    # Загружаем новую модель
    if not model_manager.load_model(request.version):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось загрузить модель",
        )

    log.info(f"Модель успешно сменена на: {request.version}")

    return ChangeModelResponse(
        success=True,
        message=f"Модель версии {request.version} успешно загружена",
        model_version=request.version,
    )


@app.get(
    "/models",
    tags=["Model Management"],
    summary="Список моделей",
    description="Получение списка всех доступных моделей",
)
async def list_models() -> list[dict]:
    """Получение списка всех доступных версий моделей."""
    if model_manager is None:
        return []

    versions = model_manager.get_available_versions()
    current = model_manager.model_version

    return [
        {"version": v, "is_current": v == current} for v in versions
    ]


@app.get("/", tags=["Root"])
async def root():
    """Корневой endpoint."""
    return {
        "service": "E-commerce Classification API",
        "version": "1.0.0",
        "status": "running",
    }


def main():
    """Точка входа для запуска через uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.inference.main:app",
        host=settings.INFERENCE_HOST,
        port=settings.INFERENCE_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
