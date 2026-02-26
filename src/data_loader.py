"""
Модуль для загрузки и предобработки данных.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загружает датасет из CSV файла.
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        DataFrame с колонками 'category' и 'description'
    """
    df = pd.read_csv(
        filepath, 
        encoding='latin-1', 
        header=None, 
        names=['category', 'description']
    )
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка данных:
    - Удаление дубликатов
    - Удаление пропусков
    - Нормализация текста (базовая)
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        Обработанный DataFrame
    """
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Удаление пропусков
    df = df.dropna(subset=['description', 'category'])
    
    # Базовая очистка текста
    df['description'] = df['description'].astype(str)
    
    return df


def split_data(
    X: pd.Series, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = True
):
    """
    Разбиение данных на train и test.
    
    Args:
        X: Признаки (тексты)
        y: Таргет (категории)
        test_size: Размер тестовой выборки
        random_state: Seed для воспроизводимости
        stratify: Использовать ли стратификацию
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
