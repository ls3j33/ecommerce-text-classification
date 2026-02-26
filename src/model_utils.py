"""
Модуль для обучения и оценки моделей.
"""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model_cv(model, X, y, cv=5, scoring='f1_macro'):
    """
    Оценка модели с помощью кросс-валидации.
    
    Args:
        model: Обучаемая модель
        X: Признаки
        y: Таргет
        cv: Количество фолдов
        scoring: Метрика для оценки
        
    Returns:
        dict с метриками
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }


def evaluate_model(model, X_test, y_test, labels=None):
    """
    Оценка модели на тестовой выборке.
    
    Args:
        model: Обученная модель
        X_test: Тестовые признаки
        y_test: Тестовый таргет
        labels: Названия классов
        
    Returns:
        dict с метриками
    """
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred
    }


def print_classification_report(y_true, y_pred, labels=None, target_names=None):
    """Вывод classification report."""
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Построение матрицы ошибок.
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
        labels: Названия классов
        title: Заголовок графика
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_cv_scores(scores_dict, title='Cross-Validation Scores'):
    """
    Визуализация результатов кросс-валидации.
    
    Args:
        scores_dict: dict {model_name: {'mean': float, 'std': float, 'scores': array}}
        title: Заголовок графика
    """
    models = list(scores_dict.keys())
    means = [scores_dict[m]['mean'] for m in models]
    stds = [scores_dict[m]['std'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.8)
    plt.ylabel('F1 Macro Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Добавление значений на столбцы
    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.02, 
            f'{mean:.3f}', 
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    plt.show()


def get_feature_importance(model, feature_names=None, top_n=20):
    """
    Извлечение важности признаков из модели.
    
    Args:
        model: Обученная модель
        feature_names: Названия признаков
        top_n: Количество топ признаков
        
    Returns:
        DataFrame с важностями
    """
    import pandas as pd
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_.mean(axis=0))
    else:
        return None
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)
