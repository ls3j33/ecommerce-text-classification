"""
Фабрика моделей для классификации текста.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np


def get_model(model_name: str, **kwargs):
    """
    Фабрика моделей для классификации текста.
    
    Args:
        model_name: Название модели
        **kwargs: Дополнительные параметры модели
    
    Returns:
        Модель sklearn
    
    Примеры:
        >>> get_model('logreg', C=1.5)
        >>> get_model('svm', C=0.5)
        >>> get_model('xgb', n_estimators=200)
    """
    models = {
        'logreg': LogisticRegression,
        'svm': LinearSVC,
        'nb': MultinomialNB,
        'rf': RandomForestClassifier,
    }
    
    # Добавляем XGBoost и LightGBM если доступны
    try:
        import xgboost as xgb
        models['xgb'] = xgb.XGBClassifier
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        models['lgb'] = lgb.LGBMClassifier
    except ImportError:
        pass
    
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )
    
    return models[model_name](**kwargs)


def get_ensemble_model(models_list, voting='soft'):
    """
    Создание ансамбля моделей (VotingClassifier).
    
    Args:
        models_list: Список кортей (name, model)
        voting: 'soft' (по вероятностям) или 'hard' (по большинству)
    
    Returns:
        VotingClassifier
    """
    return VotingClassifier(estimators=models_list, voting=voting)


def get_default_params(model_name: str):
    """
    Параметры по умолчанию для моделей.
    
    Args:
        model_name: Название модели
    
    Returns:
        dict с параметрами
    """
    defaults = {
        'logreg': {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        },
        'svm': {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'nb': {},
        'rf': {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        },
        'xgb': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        },
        'lgb': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }
    return defaults.get(model_name, {})
