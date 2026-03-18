"""Модуль для очистки текста."""

import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from src.logging_config import log

# Скачиваем ресурсы nltk при импорте
_RESOURCES_DOWNLOADED = False


def _download_nltk_resources() -> None:
    """Загрузка ресурсов NLTK один раз при импорте."""
    global _RESOURCES_DOWNLOADED
    if _RESOURCES_DOWNLOADED:
        return

    resources = [
        ('stopwords', 'корпуса стоп-слов'),
        ('wordnet', 'лексической базы WordNet'),
        ('omw-1.4', 'расширения OMW'),
        ('averaged_perceptron_tagger', 'POS-теггера'),
    ]

    for resource, desc in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            try:
                # Пробуем альтернативное имя для новых версий
                nltk.download(f'{resource}_eng', quiet=True)
            except Exception as e:
                log.warning(f"Не удалось загрузить {desc} ({resource}): {e}")

    _RESOURCES_DOWNLOADED = True
    log.info("Ресурсы NLTK загружены")


# Загружаем ресурсы при импорте модуля
_download_nltk_resources()


class TextCleaner:
    """
    Класс для очистки текста.

    Выполняет:
    - Приведение к нижнему регистру
    - Удаление HTML-тегов, URL, специальных символов
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
        lemmatize: bool = False,
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_url = remove_url
        self.remove_digits = remove_digits
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # Инициализация лемматизатора
        self._lemmatizer: Optional[WordNetLemmatizer] = None
        self._stop_words: Optional[set] = None

    def _get_lemmatizer(self) -> WordNetLemmatizer:
        """Ленивая инициализация лемматизатора."""
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    def _get_stop_words(self) -> set:
        """Ленивая инициализация стоп-слов."""
        if self._stop_words is None:
            try:
                self._stop_words = set(stopwords.words("english"))
            except Exception as e:
                log.warning(f"Не удалось загрузить стоп-слова: {e}")
                self._stop_words = set()
        return self._stop_words

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """
        Конвертация penn treebank tag в wordnet tag.

        Args:
            treebank_tag: POS тег в формате Penn Treebank

        Returns:
            POS тег в формате WordNet (n, v, a, r)
        """
        if treebank_tag.startswith("NN"):
            return "n"  # существительное
        elif treebank_tag.startswith("VB"):
            return "v"  # глагол
        elif treebank_tag.startswith("JJ"):
            return "a"  # прилагательное
        elif treebank_tag.startswith("RB"):
            return "r"  # наречие
        else:
            return "n"  # по умолчанию существительное

    def clean_text(self, text: str) -> str:
        """
        Очистка одного текста.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        if not isinstance(text, str):
            return ""

        # Приведение к нижнему регистру
        if self.lowercase:
            text = text.lower()

        # Удаление HTML тегов
        if self.remove_html:
            text = re.sub(r"<[^>]+>", "", text)

        # Удаление URL
        if self.remove_url:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Удаление упоминаний и хэштегов
        text = re.sub(r"@\w+|#\w+", "", text)

        # Удаление цифр
        if self.remove_digits:
            text = re.sub(r"\d+", "", text)

        # Удаление пунктуации
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()

        # Токенизация
        tokens = text.split()

        # Удаление стоп-слов
        if self.remove_stopwords:
            stop_words = self._get_stop_words()
            tokens = [t for t in tokens if t not in stop_words]

        # Лемматизация с использованием pos_tag
        if self.lemmatize:
            try:
                lemmatizer = self._get_lemmatizer()
                pos_tags = pos_tag(tokens)
                tokens = [
                    lemmatizer.lemmatize(t, self._get_wordnet_pos(pos))
                    for t, pos in pos_tags
                ]
            except Exception as e:
                log.warning(f"Не удалось выполнить лемматизацию: {e}")

        return " ".join(tokens)

    def clean_texts(self, texts: list[str]) -> list[str]:
        """
        Очистка списка текстов.

        Args:
            texts: Список текстов

        Returns:
            Список очищенных текстов
        """
        log.debug(f"Очистка {len(texts)} текстов")
        return [self.clean_text(text) for text in texts]

    def __call__(self, text: str) -> str:
        """Вызов объекта как функции."""
        return self.clean_text(text)
