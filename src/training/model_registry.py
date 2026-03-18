"""Регистр моделей для хранения и управления версиями моделей."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import settings
from src.logging_config import log


class ModelMetadata:
    """Метаданные модели."""

    def __init__(
        self,
        version: str,
        model_type: str,
        metrics: dict[str, float],
        created_at: str,
        path: str,
        description: str = "",
    ):
        self.version = version
        self.model_type = model_type
        self.metrics = metrics
        self.created_at = created_at
        self.path = path
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "version": self.version,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "path": self.path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Создание из словаря."""
        return cls(
            version=data["version"],
            model_type=data["model_type"],
            metrics=data["metrics"],
            created_at=data["created_at"],
            path=data["path"],
            description=data.get("description", ""),
        )


class ModelRegistry:
    """
    Регистр моделей для управления версиями.

    Хранит метаданные всех версий моделей и управляет текущей активной моделью.
    """

    REGISTRY_FILE = "registry.json"
    CURRENT_LINK = "current"

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Инициализация реестра.

        Args:
            registry_path: Путь к директории реестра моделей
        """
        self.registry_path = registry_path or settings.model_registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.registry_path / self.REGISTRY_FILE
        self._current_link = self.registry_path / self.CURRENT_LINK

        # Инициализация реестра
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Создание файла реестра, если он не существует."""
        if not self._registry_file.exists():
            self._write_registry({"models": [], "current_version": None})
            log.info(f"Создан новый реестр моделей: {self._registry_file}")

    def _read_registry(self) -> dict[str, Any]:
        """Чтение реестра из файла."""
        try:
            with open(self._registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.warning(f"Ошибка чтения реестра: {e}. Создаём новый.")
            return {"models": [], "current_version": None}

    def _write_registry(self, data: dict[str, Any]) -> None:
        """Запись реестра в файл."""
        self._registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.debug(f"Реестр обновлён: {self._registry_file}")

    def _get_next_version(self) -> str:
        """Получение следующего номера версии."""
        registry = self._read_registry()
        existing_versions = [m["version"] for m in registry["models"]]

        if not existing_versions:
            return "v1"

        # Извлекаем номера версий и находим максимальный
        version_numbers = []
        for v in existing_versions:
            if v.startswith("v"):
                try:
                    version_numbers.append(int(v[1:]))
                except ValueError:
                    continue

        if version_numbers:
            return f"v{max(version_numbers) + 1}"
        return "v1"

    def register_model(
        self,
        model_path: Path,
        model_type: str,
        metrics: dict[str, float],
        description: str = "",
        set_current: bool = True,
    ) -> ModelMetadata:
        """
        Регистрация новой модели в реестре.

        Args:
            model_path: Путь к файлам модели
            model_type: Тип модели (distilbert, svm, etc.)
            metrics: Метрики модели (f1_macro, accuracy, etc.)
            description: Описание модели
            set_current: Сделать ли эту модель текущей

        Returns:
            ModelMetadata зарегистрированной модели
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        version = self._get_next_version()
        version_dir = self.registry_path / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Копируем файлы модели в директорию версии
        if model_path.is_file():
            dest_path = version_dir / model_path.name
            shutil.copy2(model_path, dest_path)
        elif model_path.is_dir():
            dest_path = version_dir
            for item in model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)

        metadata = ModelMetadata(
            version=version,
            model_type=model_type,
            metrics=metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
            path=str(version_dir),
            description=description,
        )

        # Добавляем в реестр
        registry = self._read_registry()
        registry["models"].append(metadata.to_dict())

        if set_current:
            registry["current_version"] = version
            self._set_current(version)

        self._write_registry(registry)
        log.info(f"Зарегистрирована новая модель: {version} в {version_dir}")

        return metadata

    def _set_current(self, version: str) -> None:
        """
        Установка текущей версии модели.

        Args:
            version: Версия модели
        """
        version_dir = self.registry_path / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Версия модели не найдена: {version}")

        # Создаём symlink или файл-указатель
        if self._current_link.exists():
            if self._current_link.is_symlink() or self._current_link.is_file():
                self._current_link.unlink()

        # Создаём файл с указанием текущей версии
        self._current_link.write_text(version, encoding="utf-8")
        log.info(f"Текущая версия модели: {version}")

    def get_current_model(self) -> Optional[ModelMetadata]:
        """
        Получение метаданных текущей модели.

        Returns:
            ModelMetadata текущей модели или None
        """
        registry = self._read_registry()
        current_version = registry.get("current_version")

        if not current_version:
            return None

        for model in registry["models"]:
            if model["version"] == current_version:
                return ModelMetadata.from_dict(model)

        return None

    def get_model_by_version(self, version: str) -> Optional[ModelMetadata]:
        """
        Получение метаданных модели по версии.

        Args:
            version: Версия модели

        Returns:
            ModelMetadata модели или None
        """
        registry = self._read_registry()

        for model in registry["models"]:
            if model["version"] == version:
                return ModelMetadata.from_dict(model)

        return None

    def get_current_model_path(self) -> Optional[Path]:
        """
        Получение пути к текущей модели.

        Returns:
            Path к директории текущей модели или None
        """
        current = self.get_current_model()
        if current:
            return Path(current.path)
        return None

    def list_models(self) -> list[ModelMetadata]:
        """
        Получение списка всех моделей.

        Returns:
            Список ModelMetadata
        """
        registry = self._read_registry()
        return [ModelMetadata.from_dict(m) for m in registry["models"]]

    def validate_metrics(self, metrics: dict[str, float], min_f1: float) -> bool:
        """
        Валидация метрик модели.

        Args:
            metrics: Метрики модели
            min_f1: Минимальный порог F1-macro

        Returns:
            True если метрики проходят валидацию
        """
        f1_macro = metrics.get("f1_macro", 0)

        if f1_macro < min_f1:
            log.warning(
                f"Метрика F1-macro ({f1_macro:.4f}) ниже порога ({min_f1})"
            )
            return False

        log.info(f"Метрики прошли валидацию: F1-macro = {f1_macro:.4f}")
        return True
