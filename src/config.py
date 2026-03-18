"""Конфигурация приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Настройки приложения из .env файла."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Paths
    DATA_RAW_PATH: str = Field(default="data/raw/ecommerceDataset.csv")
    DATA_PROCESSED_PATH: str = Field(default="data/processed/")
    MODEL_REGISTRY_PATH: str = Field(default="models/registry/")

    # Model config
    MODEL_TYPE: str = Field(default="distilbert")
    MIN_F1_THRESHOLD: float = Field(default=0.95)

    # Database
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="ecommerce")
    DB_USER: str = Field(default="user")
    DB_PASSWORD: str = Field(default="password")

    # Inference
    INFERENCE_HOST: str = Field(default="0.0.0.0")
    INFERENCE_PORT: int = Field(default=8000)

    # Training
    TRAIN_BATCH_SIZE: int = Field(default=32)
    TRAIN_EPOCHS: int = Field(default=4)
    TRAIN_LEARNING_RATE: float = Field(default=5e-5)

    @property
    def db_url(self) -> str:
        """URL подключения к базе данных."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def data_raw_path(self) -> Path:
        """Путь к сырым данным."""
        return Path(self.DATA_RAW_PATH)

    @property
    def data_processed_path(self) -> Path:
        """Путь к обработанным данным."""
        return Path(self.DATA_PROCESSED_PATH)

    @property
    def model_registry_path(self) -> Path:
        """Путь к реестру моделей."""
        return Path(self.MODEL_REGISTRY_PATH)


settings = Settings()
