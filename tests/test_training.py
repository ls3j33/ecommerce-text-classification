"""Тесты для Training сервиса."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.training.validate import ModelValidator
from src.training.model_registry import ModelRegistry, ModelMetadata
from src.training.train import ModelTrainer


class TestModelValidator:
    """Тесты для ModelValidator."""

    @pytest.fixture
    def validator(self):
        """Фикстура валидатора."""
        return ModelValidator(min_f1_macro=0.8)

    def test_compute_metrics(self, validator):
        """Тест вычисления метрик."""
        y_true = np.array(["A", "B", "A", "B", "A", "B"])
        y_pred = np.array(["A", "B", "A", "B", "A", "B"])

        metrics = validator.compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_validate_success(self, validator):
        """Тест успешной валидации."""
        metrics = {"f1_macro": 0.9}
        is_valid, message = validator.validate(metrics)
        assert is_valid is True
        assert "прошла" in message.lower()

    def test_validate_failure(self, validator):
        """Тест проваленной валидации."""
        metrics = {"f1_macro": 0.5}
        is_valid, message = validator.validate(metrics)
        assert is_valid is False
        assert "не прошла" in message.lower()

    def test_classification_report(self, validator):
        """Тест отчёта классификации."""
        y_true = np.array(["A", "B", "A", "B"])
        y_pred = np.array(["A", "B", "A", "B"])

        report = validator.get_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "accuracy" in report.lower()


class TestModelRegistry:
    """Тесты для ModelRegistry."""

    @pytest.fixture
    def temp_registry(self):
        """Фикстура временного реестра."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=Path(tmpdir))
            yield registry

    def test_registry_initialization(self, temp_registry):
        """Тест инициализации реестра."""
        assert temp_registry._registry_file.exists()

    def test_register_model(self, temp_registry):
        """Тест регистрации модели."""
        # Создаём временный файл модели
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"model data")
            model_path = Path(f.name)

        try:
            metadata = temp_registry.register_model(
                model_path=model_path,
                model_type="svm",
                metrics={"f1_macro": 0.95},
                description="Test model",
            )

            assert metadata.version == "v1"
            assert metadata.model_type == "svm"
            assert metadata.metrics["f1_macro"] == 0.95
        finally:
            model_path.unlink()

    def test_get_current_model(self, temp_registry):
        """Тест получения текущей модели."""
        # Сначала регистрируем модель
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"model data")
            model_path = Path(f.name)

        try:
            temp_registry.register_model(
                model_path=model_path,
                model_type="svm",
                metrics={"f1_macro": 0.95},
            )

            current = temp_registry.get_current_model()
            assert current is not None
            assert current.version == "v1"
        finally:
            model_path.unlink()

    def test_list_models(self, temp_registry):
        """Тест списка моделей."""
        models = temp_registry.list_models()
        assert isinstance(models, list)
        assert len(models) == 0  # Пустой реестр

    def test_validate_metrics(self, temp_registry):
        """Тест валидации метрик."""
        metrics = {"f1_macro": 0.96}
        assert temp_registry.validate_metrics(metrics, min_f1=0.95) is True

        metrics_low = {"f1_macro": 0.90}
        assert temp_registry.validate_metrics(metrics_low, min_f1=0.95) is False


class TestModelTrainer:
    """Тесты для ModelTrainer."""

    @pytest.fixture
    def trainer(self):
        """Фикстура тренера."""
        return ModelTrainer(model_type="svm", max_features=100)

    def test_preprocess(self, trainer):
        """Тест препроцессинга."""
        texts = ["Hello World!", "Test <b>HTML</b>"]
        cleaned = trainer.preprocess(texts)
        assert len(cleaned) == 2
        assert all(isinstance(t, str) for t in cleaned)

    def test_preprocess_single_text(self, trainer):
        """Тест препроцессинга одного текста."""
        cleaned = trainer._text_cleaner.clean_text("Hello World!")
        assert isinstance(cleaned, str)
