# Модули проекта E-commerce Text Classification

## Структура `src/`

```
src/
├── __init__.py              # Импорты всех публичных API
├── data_loader.py           # Загрузка и предобработка данных
├── text_transformers.py     # Трансформеры текста (sklearn-совместимые)
├── embeddings.py            # Векторизация через эмбеддинги
├── feature_selection.py     # Отбор признаков
├── models.py                # Фабрика моделей
├── model_utils.py           # Утилиты для обучения и оценки
└── visualization.py         # Функции визуализации
```

---

## Модули

### `data_loader.py`
Загрузка и предобработка данных.

**Функции:**
- `load_data(filepath)` — загрузка CSV
- `preprocess_data(df)` — базовая предобработка (дубликаты, пропуски)
- `split_data(X, y, test_size, stratify)` — разбиение на train/test

**Пример:**
```python
from src.data_loader import load_data, split_data

df = load_data('data/ecommerceDataset.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df['description'], df['category'])
```

---

### `text_transformers.py`
Трансформеры текста для sklearn Pipeline.

**Классы:**
- `TextCleaner` — очистка текста (HTML, URL, стоп-слова, лемматизация)
- `TextLengthExtractor` — извлечение длины текста
- `CombinedFeatures` — объединение TF-IDF и числовых фич

**Пример:**
```python
from src.text_transformers import TextCleaner
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ('cleaner', TextCleaner(remove_html=True, remove_stopwords=True, lemmatize=True)),
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression())
])
```

---

### `embeddings.py`
Векторизация через предобученные эмбеддинги.

**Классы:**
- `SentenceTransformerVectorizer` — Sentence-BERT эмбеддинги
- `EmbeddingVectorizer` — усреднение word-эмбеддингов (GloVe, Word2Vec)

**Пример:**
```python
from src.embeddings import SentenceTransformerVectorizer

pipeline = Pipeline([
    ('embeddings', SentenceTransformerVectorizer(model_name='all-MiniLM-L6-v2')),
    ('clf', LogisticRegression())
])
```

---

### `feature_selection.py`
Отбор признаков.

**Классы и функции:**
- `FeatureSelector` — sklearn-совместимый селектор (Chi-2, Mutual Information)
- `select_features_chi2(X, y, k)` — отбор через Chi-2
- `select_features_mutual_info(X, y, k)` — отбор через Mutual Information

**Пример:**
```python
from src.feature_selection import FeatureSelector
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000)),
    ('select', FeatureSelector(method='chi2', k=5000)),
    ('clf', LogisticRegression())
])
```

---

### `models.py`
Фабрика моделей.

**Функции:**
- `get_model(model_name, **kwargs)` — создание модели по названию
- `get_ensemble_model(models_list, voting)` — создание ансамбля
- `get_default_params(model_name)` — параметры по умолчанию

**Пример:**
```python
from src.models import get_model, get_default_params

# Создание модели
model = get_model('logreg', C=1.5, class_weight='balanced')

# Или с параметрами по умолчанию
params = get_default_params('xgb')
model = get_model('xgb', **params)
```

**Доступные модели:**
- `'logreg'` — Logistic Regression
- `'svm'` — Linear SVM
- `'nb'` — Naive Bayes
- `'rf'` — Random Forest
- `'xgb'` — XGBoost (если установлен)
- `'lgb'` — LightGBM (если установлен)

---

### `model_utils.py`
Утилиты для оценки моделей.

**Функции:**
- `evaluate_model_cv(model, X, y, cv, scoring)` — кросс-валидация
- `evaluate_model(model, X_test, y_test)` — оценка на тесте
- `plot_confusion_matrix(y_true, y_pred, labels)` — матрица ошибок
- `plot_cv_scores(scores_dict)` — визуализация CV scores
- `get_feature_importance(model, feature_names, top_n)` — важность признаков

**Пример:**
```python
from src.model_utils import evaluate_model, plot_confusion_matrix

metrics = evaluate_model(model, X_test, y_test)
print(f"F1-macro: {metrics['f1_macro']:.4f}")

plot_confusion_matrix(y_test, metrics['y_pred'], labels=model.classes_)
```

---

### `visualization.py`
Визуализация для EDA.

**Функции:**
- `plot_target_distribution(y)` — распределение классов
- `plot_text_length_distribution(df, by_category)` — распределение длин текстов
- `plot_top_words(df, n_words, category)` — топ частотных слов
- `plot_word_cloud(texts)` — word cloud (требует wordcloud)

**Пример:**
```python
from src.visualization import plot_target_distribution, plot_top_words

plot_target_distribution(df['category'])
plot_top_words(df, n_words=20, category='Books')
```

---

## Полный пример использования

```python
import sys
sys.path.append('src')

from src.data_loader import load_data, preprocess_data, split_data
from src.text_transformers import TextCleaner
from src.feature_selection import FeatureSelector
from src.models import get_model
from src.model_utils import evaluate_model, plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка данных
df = load_data('data/ecommerceDataset.csv')
df = preprocess_data(df)

# Разбиение
X_train, X_test, y_train, y_test = split_data(df['description'], df['category'])

# Пайплайн с отбором признаков
pipeline = Pipeline([
    ('cleaner', TextCleaner(remove_html=True, remove_stopwords=True, lemmatize=True)),
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
    ('select', FeatureSelector(method='chi2', k=5000)),
    ('clf', get_model('logreg', C=1.5, class_weight='balanced'))
])

# Обучение
pipeline.fit(X_train, y_train)

# Оценка
metrics = evaluate_model(pipeline, X_test, y_test)
print(f"F1-macro: {metrics['f1_macro']:.4f}")

# Визуализация
plot_confusion_matrix(y_test, metrics['y_pred'], labels=pipeline.classes_)
```
