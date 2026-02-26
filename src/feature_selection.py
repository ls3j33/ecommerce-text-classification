"""
Модуль для отбора признаков.
"""
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Отбор признаков через Chi-2 или Mutual Information.
    
    Args:
        method: Метод отбора ('chi2' или 'mutual_info')
        k: Количество признаков для отбора
    """
    
    def __init__(self, method: str = 'chi2', k: int = 5000):
        self.method = method
        self.k = k
        self.selector = None
    
    def fit(self, X, y):
        """Обучение селектора."""
        if self.method == 'chi2':
            self.selector = SelectKBest(score_func=chi2, k=self.k)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'chi2' or 'mutual_info'.")
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        """Отбор признаков."""
        return self.selector.transform(X)
    
    def get_support(self, indices=False):
        """Получить маску выбранных признаков."""
        return self.selector.get_support(indices=indices)
    
    def get_feature_names_out(self, input_features=None):
        """Названия выбранных признаков."""
        if input_features is None:
            input_features = [f'feature_{i}' for i in range(X.shape[1])]
        
        selected_indices = self.get_support(indices=True)
        return [input_features[i] for i in selected_indices]


def select_features_chi2(X, y, k=5000):
    """
    Отбор признаков через Chi-2.
    
    Args:
        X: Признаки (sparse matrix или array)
        y: Таргет
        k: Количество признаков для отбора
    
    Returns:
        X_selected, selector
    """
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


def select_features_mutual_info(X, y, k=5000):
    """
    Отбор признаков через Mutual Information.
    
    Args:
        X: Признаки (sparse matrix или array)
        y: Таргет
        k: Количество признаков для отбора
    
    Returns:
        X_selected, selector
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector
