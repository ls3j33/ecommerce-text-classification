"""Настройка логирования для приложения."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """
    Настройка логгера.

    Args:
        log_file: Путь к файлу логов. Если None, логи пишутся только в консоль.
        level: Уровень логирования.
        format_string: Формат сообщения лога.
    """
    # Удаляем стандартный обработчик
    logger.remove()

    # Формат по умолчанию
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Добавляем обработчик консоли
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )

    # Добавляем обработчик файла, если указан
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    logger.info("Логгер инициализирован")


# Создаем стандартный логгер для использования в модулях
log = logger
