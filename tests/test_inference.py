"""Тесты для Inference сервиса."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.inference.main import app
from src.inference.model_loader import ModelManager


@pytest.fixture
def client():
    """Фикстура тестового клиента."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_model_manager():
    """Фикстура с мокнутым менеджером моделей."""
    manager = MagicMock(spec=ModelManager)
    manager.is_loaded = True
    manager.model_version = "v1"
    manager.predict.return_value = ("Electronics", 0.95)
    manager.get_available_versions.return_value = ["v1", "v2"]
    return manager


class TestHealthEndpoint:
    """Тесты health check endpoint."""

    def test_health_check(self, client):
        """Тест проверки работоспособности."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_response_structure(self, client):
        """Тест структуры ответа health."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Тесты endpoint предсказания."""

    def test_predict_success(self, client, mock_model_manager):
        """Тест успешного предсказания."""
        with patch("src.inference.main.model_manager", mock_model_manager):
            response = client.post(
                "/predict",
                json={"description": "Test product description"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "category" in data
            assert "confidence" in data
            assert "model_version" in data
            assert data["category"] == "Electronics"
            assert data["confidence"] == 0.95

    def test_predict_empty_description(self, client):
        """Тест пустого описания."""
        response = client.post("/predict", json={"description": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_missing_description(self, client):
        """Тест отсутствия поля description."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_long_description(self, client):
        """Тест очень длинного описания."""
        long_text = "A" * 60000  # Больше max_length
        response = client.post("/predict", json={"description": long_text})
        assert response.status_code == 422

    def test_predict_response_structure(self, client, mock_model_manager):
        """Тест структуры ответа предсказания."""
        with patch("src.inference.main.model_manager", mock_model_manager):
            response = client.post(
                "/predict",
                json={"description": "Test product"}
            )
            data = response.json()
            assert isinstance(data["category"], str)
            assert isinstance(data["confidence"], float)
            assert 0 <= data["confidence"] <= 1


class TestChangeModelEndpoint:
    """Тесты endpoint смены модели."""

    def test_change_model_success(self, client, mock_model_manager):
        """Тест успешной смены модели."""
        mock_model_manager.load_model.return_value = True
        with patch("src.inference.main.model_manager", mock_model_manager):
            response = client.post(
                "/change-model",
                json={"version": "v2"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["model_version"] == "v2"

    def test_change_model_not_found(self, client, mock_model_manager):
        """Тест модели не найдена."""
        mock_model_manager.load_model.return_value = False
        with patch("src.inference.main.model_manager", mock_model_manager):
            response = client.post(
                "/change-model",
                json={"version": "v99"}
            )
            assert response.status_code == 404

    def test_change_model_invalid_version_format(self, client):
        """Тест неверного формата версии."""
        response = client.post(
            "/change-model",
            json={"version": "invalid"}
        )
        assert response.status_code == 422


class TestListModelsEndpoint:
    """Тесты endpoint списка моделей."""

    def test_list_models(self, client, mock_model_manager):
        """Тест списка моделей."""
        with patch("src.inference.main.model_manager", mock_model_manager):
            response = client.get("/models")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestRootEndpoint:
    """Тесты корневого endpoint."""

    def test_root(self, client):
        """Тест корневого endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "E-commerce Classification API"
