"""Модуль валидации метрик модели."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import numpy as np

from src.logging_config import log


class ModelValidator:
    """
    Валидатор метрик модели.

    Вычисляет и проверяет метрики качества модели.
    """

    def __init__(self, min_f1_macro: float = 0.95):
        """
        Инициализация валидатора.

        Args:
            min_f1_macro: Минимальный порог F1-macro для валидации
        """
        self.min_f1_macro = min_f1_macro

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Вычисление метрик модели.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки

        Returns:
            Словарь с метриками
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

        log.info(
            f"Метрики модели: "
            f"F1-macro={metrics['f1_macro']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

        return metrics

    def validate(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """
        Валидация метрик модели.

        Args:
            metrics: Словарь с метриками

        Returns:
            Tuple (успешность, сообщение)
        """
        f1_macro = metrics.get("f1_macro", 0)

        if f1_macro < self.min_f1_macro:
            msg = (
                f"Модель не прошла валидацию: "
                f"F1-macro={f1_macro:.4f} < порога={self.min_f1_macro}"
            )
            log.warning(msg)
            return False, msg

        msg = (
            f"Модель прошла валидацию: "
            f"F1-macro={f1_macro:.4f} >= порога={self.min_f1_macro}"
        )
        log.info(msg)
        return True, msg

    def get_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """
        Получение отчёта классификации.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки

        Returns:
            Текстовый отчёт
        """
        return classification_report(y_true, y_pred, zero_division=0)
