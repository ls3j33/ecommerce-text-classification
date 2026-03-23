"""Модуль загрузки и управления моделями для Inference."""

import pickle
from pathlib import Path
from typing import Any, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings
from src.logging_config import log
from src.training.model_registry import ModelRegistry


class ModelManager:
    """
    Менеджер моделей для Inference сервиса.

    Отвечает за загрузку, выгрузку и смену моделей.
    """

    def __init__(self):
        self._model: Optional[Any] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_version: Optional[str] = None
        self._model_type: Optional[str] = None
        self._registry = ModelRegistry()
        self._is_transformer: bool = False

    @property
    def model(self) -> Optional[Any]:
        """Получение текущей модели."""
        return self._model

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        """Получение текущего токенизатора."""
        return self._tokenizer

    @property
    def model_version(self) -> Optional[str]:
        """Получение версии текущей модели."""
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель."""
        return self._model is not None

    def load_model(self, version: Optional[str] = None) -> bool:
        """
        Загрузка модели из реестра.

        Args:
            version: Версия модели для загрузки. Если None, загружается текущая.

        Returns:
            True если загрузка успешна
        """
        try:
            # Получаем метаданные модели
            if version:
                metadata = self._registry.get_model_by_version(version)
            else:
                metadata = self._registry.get_current_model()

            if metadata is None:
                log.error("Модель не найдена в реестре")
                return False

            model_path = Path(metadata.path)
            if not model_path.exists():
                log.error(f"Путь к модели не найден: {model_path}")
                return False

            log.info(f"Загрузка модели {metadata.version} из {model_path}")

            # Выгружаем предыдущую модель
            self.unload_model()

            # Определяем тип модели и загружаем
            self._model_type = metadata.model_type

            if metadata.model_type.lower() in ["distilbert", "bert", "transformer"]:
                self._load_transformer_model(model_path)
            else:
                self._load_sklearn_model(model_path)

            self._model_version = metadata.version
            log.info(f"Модель {metadata.version} успешно загружена")

            return True

        except Exception as e:
            log.error(f"Ошибка загрузки модели: {e}")
            self.unload_model()
            return False

    def _load_transformer_model(self, model_path: Path) -> None:
        """Загрузка transformer модели (DistilBERT, BERT)."""
        log.info("Загрузка transformer модели...")

        # Находим директорию с файлами модели
        model_dir = model_path
        if model_path.is_file():
            model_dir = model_path.parent

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Переводим модель в режим eval и на соответствующее устройство
        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        self._is_transformer = True
        log.info(f"Transformer модель загружена на {device}")

    def _load_sklearn_model(self, model_path: Path) -> None:
        """Загрузка sklearn модели (SVM, LogisticRegression, etc.)."""
        log.info("Загрузка sklearn модели...")

        # Ищем файл модели
        pkl_files = list(model_path.glob("*.pkl"))
        joblib_files = list(model_path.glob("*.joblib"))

        model_file = None
        if pkl_files:
            model_file = pkl_files[0]
        elif joblib_files:
            model_file = joblib_files[0]
        else:
            # Пробуем саму директорию как файл
            if model_path.is_file():
                model_file = model_path

        if model_file is None or not model_file.exists():
            raise FileNotFoundError(
                f"Файл модели не найден в {model_path}"
            )

        log.info(f"Загрузка модели из файла: {model_file}")

        with open(model_file, "rb") as f:
            self._model = pickle.load(f)

        self._is_transformer = False
        log.info("Sklearn модель загружена")

    def unload_model(self) -> None:
        """Выгрузка модели из памяти."""
        if self._model is not None:
            if self._is_transformer and hasattr(self._model, "to"):
                # Очищаем CUDA память для transformer моделей
                import torch

                self._model.to("cpu")
                del self._model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                del self._model

            self._model = None
            self._tokenizer = None
            self._model_version = None
            self._model_type = None
            self._is_transformer = False

            log.info("Модель выгружена из памяти")

    def predict(self, text: str) -> tuple[str, float]:
        """
        Предсказание категории товара.

        Args:
            text: Текст описания товара

        Returns:
            Tuple (категория, уверенность)

        Raises:
            RuntimeError: Если модель не загружена
        """
        if self._model is None:
            raise RuntimeError("Модель не загружена")

        if self._is_transformer:
            return self._predict_transformer(text)
        else:
            return self._predict_sklearn(text)

    def _predict_transformer(self, text: str) -> tuple[str, float]:
        """Предсказание transformer моделью."""
        import torch
        from src.preprocessing.text_cleaner import TextCleaner

        # ✅ Очистка текста (идентично training)
        cleaner = TextCleaner(
            lowercase=True,
            remove_html=True,
            remove_url=True,
            remove_digits=True,
            remove_punctuation=True,
            remove_stopwords=False,
            lemmatize=False,
        )
        text_cleaned = cleaner.clean_text(text)

        # Токенизация
        inputs = self._tokenizer(
            text_cleaned,  # ✅ Очищенный текст
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Перенос на устройство
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Предсказание
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probabilities, dim=-1)

        # Получение названия категории
        category_id = predicted.item()
        confidence_value = confidence.item()

        # Получаем название категории из модели
        if hasattr(self._model, "config") and hasattr(
            self._model.config, "id2label"
        ):
            category = self._model.config.id2label[category_id]
        else:
            # Fallback: используем числовое значение
            category = str(category_id)

        return category, confidence_value

    def _predict_sklearn(self, text: str) -> tuple[str, float]:
        """Предсказание sklearn моделью."""
        # Предсказание
        prediction = self._model.predict([text])
        category = prediction[0]

        # Получение уверенности
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba([text])
            confidence = float(probabilities.max())
        else:
            confidence = 1.0

        return category, confidence

    def get_available_versions(self) -> list[str]:
        """Получение списка доступных версий моделей."""
        models = self._registry.list_models()
        return [m.version for m in models]
