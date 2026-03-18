"""Модуль деплоя моделей в реестр."""

import shutil
from pathlib import Path
from typing import Any

from src.config import settings
from src.logging_config import log
from src.training.model_registry import ModelRegistry
from src.training.validate import ModelValidator


class ModelDeployer:
    """
    Деплойер моделей в реестр.

    Отвечает за:
    - Валидацию метрик модели
    - Регистрацию модели в реестре
    - Обновление текущей версии
    """

    def __init__(
        self,
        min_f1_threshold: float = None,
        registry_path: Path = None,
    ):
        """
        Инициализация деплойера.

        Args:
            min_f1_threshold: Минимальный порог F1-macro
            registry_path: Путь к реестру моделей
        """
        self.min_f1_threshold = min_f1_threshold or settings.MIN_F1_THRESHOLD
        self._registry = ModelRegistry(registry_path)
        self._validator = ModelValidator(self.min_f1_threshold)

    def deploy(
        self,
        model_path: Path,
        model_type: str,
        metrics: dict[str, float],
        description: str = "",
    ) -> bool:
        """
        Деплой модели в реестр.

        Args:
            model_path: Путь к файлам модели
            model_type: Тип модели (svm, distilbert, etc.)
            metrics: Метрики модели
            description: Описание модели

        Returns:
            True если деплой успешен
        """
        log.info(f"Начало деплоя модели из {model_path}")

        # Валидация метрик
        is_valid, message = self._validator.validate(metrics)
        if not is_valid:
            log.error(f"Деплой отклонён: {message}")
            return False

        try:
            # Регистрация модели
            metadata = self._registry.register_model(
                model_path=model_path,
                model_type=model_type,
                metrics=metrics,
                description=description,
                set_current=True,
            )

            log.info(
                f"Модель успешно зарегистрирована: "
                f"{metadata.version} с метриками F1-macro={metrics.get('f1_macro', 0):.4f}"
            )

            return True

        except Exception as e:
            log.error(f"Ошибка при деплое модели: {e}")
            return False

    def get_current_model_info(self) -> dict[str, Any] | None:
        """
        Получение информации о текущей модели.

        Returns:
            Информация о модели или None
        """
        metadata = self._registry.get_current_model()
        if metadata:
            return {
                "version": metadata.version,
                "model_type": metadata.model_type,
                "metrics": metadata.metrics,
                "created_at": metadata.created_at,
                "path": metadata.path,
            }
        return None

    def list_all_models(self) -> list[dict[str, Any]]:
        """
        Получение списка всех моделей в реестре.

        Returns:
            Список моделей
        """
        models = self._registry.list_models()
        return [
            {
                "version": m.version,
                "model_type": m.model_type,
                "metrics": m.metrics,
                "created_at": m.created_at,
            }
            for m in models
        ]
