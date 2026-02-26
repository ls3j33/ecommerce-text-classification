"""
Модуль для работы с эмбеддингами текста.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """
    Векторизация текста через Sentence Transformers.
    
    Args:
        model_name: Название модели (например, 'all-MiniLM-L6-v2')
        batch_size: Размер батча для кодирования
        show_progress: Показывать ли прогресс-бар
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        batch_size: int = 32,
        show_progress: bool = False
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = None
    
    def fit(self, X, y=None):
        """Загрузка модели."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        return self
    
    def transform(self, X):
        """Генерация эмбеддингов."""
        import pandas as pd
        texts = X.tolist() if isinstance(X, (pd.Series, list)) else X
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True
        )
    
    def get_feature_names_out(self, input_features=None):
        """Названия признаков (размерность эмбеддингов)."""
        return [f'embedding_{i}' for i in range(self.model.get_sentence_embedding_dimension())]


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Векторизация через усреднение word-эмбеддингов (GloVe, Word2Vec).
    
    Args:
        word_vectors: Предобученные векторы слов (gensim KeyedVectors)
    """
    
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        self.embedding_dim = word_vectors.vector_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Усреднение векторов слов для каждого текста."""
        import pandas as pd
        texts = X.tolist() if isinstance(X, (pd.Series, list)) else X
        result = []
        for text in texts:
            tokens = text.lower().split()
            vectors = [
                self.word_vectors[t] 
                for t in tokens 
                if t in self.word_vectors
            ]
            if vectors:
                result.append(np.mean(vectors, axis=0))
            else:
                result.append(np.zeros(self.embedding_dim))
        return np.array(result)
    
    def get_feature_names_out(self, input_features=None):
        return [f'embedding_{i}' for i in range(self.embedding_dim)]
