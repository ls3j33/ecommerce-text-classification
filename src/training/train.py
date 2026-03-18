"""Модуль обучения моделей."""

import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from src.config import settings
from src.logging_config import log
from src.preprocessing.text_cleaner import TextCleaner
from src.training.validate import ModelValidator
from src.training.model_registry import ModelRegistry


class ModelTrainer:
    """
    Тренер моделей для e-commerce классификации.

    Поддерживает:
    - Linear SVM с TF-IDF
    - DistilBERT (fine-tuning)
    - Logistic Regression с TF-IDF
    """

    def __init__(
        self,
        model_type: str = "svm",
        batch_size: int = 32,
        num_epochs: int = 4,
        learning_rate: float = 5e-5,
        max_features: int = 15000,
    ):
        """
        Инициализация тренера.

        Args:
            model_type: Тип модели (svm, distilbert, logistic)
            batch_size: Размер батча для transformer моделей
            num_epochs: Количество эпох обучения
            learning_rate: Learning rate для transformer моделей
            max_features: Максимальное количество признаков для TF-IDF
        """
        self.model_type = model_type.lower()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_features = max_features

        self._model: Optional[Any] = None
        self._text_cleaner = TextCleaner(
            lowercase=True,
            remove_html=True,
            remove_url=True,
            remove_digits=True,
            remove_punctuation=True,
            remove_stopwords=False,
            lemmatize=False,
        )

    def load_data(
        self, data_path: Path
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка данных для обучения.

        Args:
            data_path: Путь к CSV файлу с данными

        Returns:
            Tuple (DataFrame с данными, Series с таргетом)
        """
        log.info(f"Загрузка данных из {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")

        df = pd.read_csv(
            data_path,
            encoding="latin-1",
            header=None,
            names=["category", "description"],
        )

        # Базовая предобработка
        df = df.drop_duplicates()
        df = df.dropna(subset=["description", "category"])
        df["description"] = df["description"].astype(str)

        X = df["description"]
        y = df["category"]

        log.info(f"Загружено {len(df)} записей")
        log.info(f"Распределение классов:\n{y.value_counts()}")

        return df, y

    def preprocess(self, texts: list[str]) -> list[str]:
        """
        Препроцессинг текстов.

        Args:
            texts: Список текстов

        Returns:
            Список очищенных текстов
        """
        log.debug(f"Препроцессинг {len(texts)} текстов")
        return self._text_cleaner.clean_texts(texts)

    def train(
        self,
        X_train: list[str],
        y_train: list[str],
        X_val: Optional[list[str]] = None,
        y_val: Optional[list[str]] = None,
    ) -> Any:
        """
        Обучение модели.

        Args:
            X_train: Обучающие тексты
            y_train: Обучающие метки
            X_val: Валидационные тексты (опционально)
            y_val: Валидационные метки (опционально)

        Returns:
            Обученная модель
        """
        log.info(f"Начало обучения модели типа: {self.model_type}")

        if self.model_type == "svm":
            self._model = self._train_svm(X_train, y_train)
        elif self.model_type == "distilbert":
            self._model = self._train_distilbert(X_train, y_train, X_val, y_val)
        elif self.model_type == "logistic":
            self._model = self._train_logistic(X_train, y_train)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

        log.info("Обучение модели завершено")
        return self._model

    def _train_svm(
        self, X_train: list[str], y_train: list[str]
    ) -> Pipeline:
        """Обучение Linear SVM модели."""
        log.info("Обучение Linear SVM...")

        # Препроцессинг
        X_train_cleaned = self.preprocess(X_train)

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=(1, 1),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train_cleaned, y_train)
        log.info("Linear SVM обучена")

        return model

    def _train_logistic(
        self, X_train: list[str], y_train: list[str]
    ) -> Pipeline:
        """Обучение Logistic Regression модели."""
        log.info("Обучение Logistic Regression...")

        from sklearn.linear_model import LogisticRegression

        X_train_cleaned = self.preprocess(X_train)

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=5.0,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                        solver="lbfgs",
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

        model.fit(X_train_cleaned, y_train)
        log.info("Logistic Regression обучена")

        return model

    def _train_distilbert(
        self,
        X_train: list[str],
        y_train: list[str],
        X_val: Optional[list[str]] = None,
        y_val: Optional[list[str]] = None,
    ) -> Any:
        """Обучение DistilBERT модели."""
        log.info("Обучение DistilBERT...")

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from sklearn.preprocessing import LabelEncoder

        # Препроцессинг
        X_train_cleaned = self.preprocess(X_train)

        # Кодирование меток
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Токенизация
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_encodings = tokenizer(
            X_train_cleaned,
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Валидационные данные
        val_encodings = None
        if X_val and y_val:
            X_val_cleaned = self.preprocess(X_val)
            y_val_encoded = label_encoder.transform(y_val)
            val_encodings = tokenizer(
                X_val_cleaned,
                truncation=True,
                padding=True,
                max_length=512,
            )

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()
                }
                if self.labels:
                    item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_dataset = Dataset(train_encodings, y_train_encoded)
        val_dataset = (
            Dataset(val_encodings, label_encoder.transform(y_val))
            if val_encodings
            else None
        )

        # Модель
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_encoder.classes_),
            id2label={i: label for i, label in enumerate(label_encoder.classes_)},
            label2id={label: i for i, label in enumerate(label_encoder.classes_)},
        )

        # Параметры обучения
        training_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
        )

        # Метрика
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "f1": float(f1_score(
                    labels, predictions, average="macro", zero_division=0
                ))
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Обучение
        trainer.train()

        # Сохраняем модель и токенизатор вместе с label_encoder
        save_dir = Path(tempfile.mkdtemp())
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        # Сохраняем label_encoder
        with open(save_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        log.info(f"DistilBERT обучен, сохранён в {save_dir}")

        return save_dir

    def predict(self, X: list[str]) -> list[str]:
        """
        Предсказание модели.

        Args:
            X: Список текстов

        Returns:
            Список предсказаний
        """
        if self._model is None:
            raise RuntimeError("Модель не обучена")

        X_cleaned = self.preprocess(X)

        if self.model_type == "distilbert":
            # Для DistilBERT используем forward pass
            import torch
            from transformers import AutoTokenizer

            save_dir = self._model
            tokenizer = AutoTokenizer.from_pretrained(str(save_dir))

            encodings = tokenizer(
                X_cleaned,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(
                str(save_dir)
            )
            model.eval()

            with torch.no_grad():
                outputs = model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=-1)

            # Загружаем label_encoder
            with open(save_dir / "label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)

            return label_encoder.inverse_transform(predictions.numpy())
        else:
            return list(self._model.predict(X_cleaned))

    def save_model(self, path: Path) -> None:
        """
        Сохранение модели.

        Args:
            path: Путь для сохранения
        """
        if self._model is None:
            raise RuntimeError("Модель не обучена")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model_type == "distilbert":
            # Для DistilBERT копируем файлы из временной директории
            import shutil

            source_dir = self._model
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, path / item.name, dirs_exist_ok=True)
        else:
            # Для sklearn моделей используем pickle
            model_file = path / "model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(self._model, f)

        log.info(f"Модель сохранена в {path}")
