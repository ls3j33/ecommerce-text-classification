"""
Модуль для визуализации в EDA и оценке моделей.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_target_distribution(y, title='Target Distribution'):
    """
    Построение графика распределения целевой переменной.

    Args:
        y: Серия с категориями
        title: Заголовок графика
    """
    plt.figure(figsize=(10, 6))
    counts = y.value_counts()
    bars = plt.bar(counts.index, counts.values, alpha=0.8)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title(title)

    # Добавление значений на столбцы
    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 100,
            str(count),
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.show()


def plot_text_length_distribution(df, text_column='description', by_category=True):
    """
    Построение графика распределения длин текстов.

    Args:
        df: DataFrame с данными
        text_column: Название колонки с текстом
        by_category: Группировать ли по категориям
    """
    df['text_length'] = df[text_column].str.len()

    if by_category:
        plt.figure(figsize=(12, 6))
        categories = df['category'].unique()
        for cat in categories:
            subset = df[df['category'] == cat]['text_length']
            sns.kdeplot(subset, label=cat, fill=True, alpha=0.3)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Density')
        plt.title('Text Length Distribution by Category')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=50, kde=True)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Count')
        plt.title('Text Length Distribution')
        plt.tight_layout()
        plt.show()


def plot_text_length_stats(df, text_column='description'):
    """
    Статистика длин текстов по категориям.

    Args:
        df: DataFrame с данными
        text_column: Название колонки с текстом
    """
    df['text_length'] = df[text_column].str.len()
    stats = df.groupby('category')['text_length'].agg(['count', 'mean', 'median', 'min', 'max'])
    print(stats.round(2))


def plot_word_cloud(texts, title='Word Cloud', max_words=100):
    """
    Построение word cloud (требует wordcloud).

    Args:
        texts: Список текстов или серия
        title: Заголовок
        max_words: Максимальное количество слов
    """
    try:
        from wordcloud import WordCloud

        all_text = ' '.join(texts.astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color='white'
        ).generate(all_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Установите wordcloud: pip install wordcloud")


def plot_top_words(df, text_column='description', n_words=20, category=None):
    """
    Построение графика топ-N частотных слов.

    Args:
        df: DataFrame с данными
        text_column: Название колонки с текстом
        n_words: Количество топ слов
        category: Фильтр по категории (опционально)
    """
    from collections import Counter

    if category:
        texts = df[df['category'] == category][text_column]
    else:
        texts = df[text_column]

    # Токенизация и подсчет
    all_words = ' '.join(texts.astype(str)).lower().split()
    word_counts = Counter(all_words)

    # Фильтрация коротких слов
    word_counts = {w: c for w, c in word_counts.items() if len(w) > 3}

    top_words = word_counts.most_common(n_words)
    words, counts = zip(*top_words)

    plt.figure(figsize=(12, 6))
    plt.barh(words, counts)
    plt.xlabel('Count')
    plt.title(f'Top {n_words} Words' + (f' - {category}' if category else ''))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


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


def plot_model_comparison(results_dict, metrics=None, title='Сравнение моделей'):
    """
    Визуализация сравнения моделей по нескольким метрикам.

    Args:
        results_dict: dict {model_name: {'metric1': value, 'metric2': value, ...}}
        metrics: список метрик для отображения (по умолчанию все)
        title: Заголовок графика
    """
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())

    df_results = pd.DataFrame(results_dict).T

    plt.figure(figsize=(14, 6))
    x = np.arange(len(df_results))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        plt.bar(x + offset, df_results[metric], width, label=metric, alpha=0.8)

    plt.xlabel('Модель')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, df_results.index, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
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


def plot_feature_importance(importance_df, top_n=20, title='Важность признаков'):
    """
    Визуализация важности признаков.

    Args:
        importance_df: DataFrame с колонками 'feature' и 'importance'
        top_n: Количество топ признаков
        title: Заголовок графика
    """
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Важность')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_metrics_history(history_dict, metrics=None, title='История метрик'):
    """
    Визуализация истории метрик (для Optuna или обучения).

    Args:
        history_dict: dict {metric_name: [values]}
        metrics: список метрик для отображения
        title: Заголовок графика
    """
    if metrics is None:
        metrics = list(history_dict.keys())

    plt.figure(figsize=(12, 6))
    for metric in metrics:
        if metric in history_dict:
            plt.plot(history_dict[metric], label=metric, marker='o', markersize=3)

    plt.xlabel('Iteration / Epoch')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
