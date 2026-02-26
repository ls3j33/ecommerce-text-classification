"""
Модуль для текстовых трансформеров.
Все трансформеры совместимы со sklearn Pipeline.
"""
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Трансформер для очистки текста.

    Выполняет:
    - Приведение к нижнему регистру
    - Удаление HTML-тегов, URL, специальных символов, цифр, лишних пробелов
    - Удаление стоп-слов (опционально)
    - Лемматизацию (опционально)
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_url: bool = True,
        remove_digits: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_url = remove_url
        self.remove_digits = remove_digits
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        """Трансформер не требует обучения."""
        if self.remove_stopwords or self.lemmatize:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except:
                pass
        return self

    def transform(self, X):
        """Очистка текстов."""
        if isinstance(X, list):
            return [self._clean_text(text) for text in X]
        return X.apply(self._clean_text)

    def _clean_text(self, text: str) -> str:
        """Очистка одного текста."""
        if not isinstance(text, str):
            return ""

        # Приведение к нижнему регистру
        if self.lowercase:
            text = text.lower()

        # Удаление HTML тегов
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)

        # Удаление URL
        if self.remove_url:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Удаление упоминаний и хэштегов
        text = re.sub(r'@\w+|#\w+', '', text)

        # Удаление цифр
        if self.remove_digits:
            text = re.sub(r'\d+', '', text)

        # Удаление пунктуации
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        # Токенизация
        tokens = text.split()

        # Удаление стоп-слов
        if self.remove_stopwords:
            try:
                stop_words = set(stopwords.words('english'))
                tokens = [t for t in tokens if t not in stop_words]
            except:
                pass

        # Лемматизация
        if self.lemmatize:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            except:
                pass

        return ' '.join(tokens)

    def get_feature_names_out(self, input_features=None):
        """Для совместимости с Pipeline."""
        return ['description']


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Трансформер для извлечения длины текста.
    Может использоваться как дополнительная фича.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Извлечение длины текста."""
        if isinstance(X, list):
            return np.array([len(text) for text in X]).reshape(-1, 1)
        return X.str.len().values.reshape(-1, 1)


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """
    Объединение TF-IDF и числовых фич (например, длина текста).
    """

    def __init__(self, tfidf_vectorizer, text_cleaner=None):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.text_cleaner = text_cleaner
        self.len_mean = None  # Статистика для нормализации
        self.len_std = None

    def fit(self, X, y=None):
        if self.text_cleaner:
            self.text_cleaner.fit(X, y)
            X_clean = self.text_cleaner.transform(X)
        else:
            X_clean = X
        self.tfidf_vectorizer.fit(X_clean)
        
        # Вычислить и сохранить статистику для нормализации длины
        X_len = np.array([len(text) for text in X_clean]).reshape(-1, 1)
        self.len_mean = X_len.mean()
        self.len_std = X_len.std()
        return self

    def transform(self, X):
        if self.text_cleaner:
            X_clean = self.text_cleaner.transform(X)
        else:
            X_clean = X
        X_tfidf = self.tfidf_vectorizer.transform(X_clean)
        X_len = np.array([len(text) for text in X_clean]).reshape(-1, 1)
        # Нормализация с использованием статистики из fit()
        X_len = (X_len - self.len_mean) / self.len_std
        from scipy.sparse import hstack
        return hstack([X_tfidf, X_len])

    def get_feature_names_out(self, input_features=None):
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out(input_features)
        return list(tfidf_features) + ['text_length']
