"""Скрипт обучения модели для запуска в Docker контейнере."""

import sys
from pathlib import Path

from src.config import settings
from src.logging_config import setup_logger, log
from src.training.train import ModelTrainer
from src.training.validate import ModelValidator
from src.training.deploy import ModelDeployer


def main():
    """Основная функция обучения."""
    # Настройка логгера
    setup_logger(level="INFO")

    log.info("=" * 60)
    log.info("Запуск Training сервиса")
    log.info("=" * 60)

    try:
        # Загрузка данных
        data_path = settings.data_raw_path
        if not data_path.exists():
            log.error(f"Файл данных не найден: {data_path}")
            sys.exit(1)

        # Инициализация тренера
        trainer = ModelTrainer(
            model_type=settings.MODEL_TYPE,
            batch_size=settings.TRAIN_BATCH_SIZE,
            num_epochs=settings.TRAIN_EPOCHS,
            learning_rate=settings.TRAIN_LEARNING_RATE,
        )

        # Загрузка данных
        df, y = trainer.load_data(data_path)

        # Разбиение на train/val/test
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            df["description"].tolist(),
            y.tolist(),
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Ещё раз делим train на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

        log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Обучение модели
        trainer.train(X_train, y_train, X_val, y_val)

        # Предсказания на тесте
        y_pred = trainer.predict(X_test)

        # Вычисление метрик
        validator = ModelValidator()
        metrics = validator.compute_metrics(
            np.array(y_test),
            np.array(y_pred),
        )

        log.info(f"Финальные метрики на тесте: {metrics}")

        # Сохранение модели во временную директорию
        import tempfile

        temp_model_path = Path(tempfile.mkdtemp()) / "model"
        trainer.save_model(temp_model_path)

        # Деплой модели в реестр
        deployer = ModelDeployer()

        success = deployer.deploy(
            model_path=temp_model_path,
            model_type=settings.MODEL_TYPE,
            metrics=metrics,
            description=f"Модель {settings.MODEL_TYPE}, обученная на {len(df)} записях",
        )

        if success:
            log.info("Обучение и деплой успешно завершены")
            log.info("=" * 60)
            sys.exit(0)
        else:
            log.error("Деплой модели не удался")
            log.error("=" * 60)
            sys.exit(1)

    except Exception as e:
        log.error(f"Ошибка при обучении: {e}", exc_info=True)
        log.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    import numpy as np

    main()
