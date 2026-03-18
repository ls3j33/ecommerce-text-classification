"""Тесты для модуля препроцессинга текста."""

import pytest
from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Тесты для TextCleaner."""

    @pytest.fixture
    def cleaner(self):
        """Фикстура с базовым очистителем."""
        return TextCleaner()

    @pytest.fixture
    def full_cleaner(self):
        """Фикстура с полным очистителем."""
        return TextCleaner(
            lowercase=True,
            remove_html=True,
            remove_url=True,
            remove_digits=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lemmatize=True,
        )

    def test_lowercase(self, cleaner):
        """Тест приведения к нижнему регистру."""
        cleaner.lowercase = True
        assert cleaner.clean_text("Hello WORLD") == "hello world"

    def test_remove_html(self, cleaner):
        """Тест удаления HTML тегов."""
        cleaner.remove_html = True
        assert cleaner.clean_text("<p>Hello</p> <br> World") == "hello world"

    def test_remove_url(self, cleaner):
        """Тест удаления URL."""
        cleaner.remove_url = True
        result = cleaner.clean_text("Visit https://example.com for more")
        assert "https" not in result
        assert "example.com" not in result

    def test_remove_digits(self, cleaner):
        """Тест удаления цифр."""
        cleaner.remove_digits = True
        result = cleaner.clean_text("Product 123 costs $456")
        assert not any(c.isdigit() for c in result)

    def test_remove_punctuation(self, cleaner):
        """Тест удаления пунктуации."""
        cleaner.remove_punctuation = True
        result = cleaner.clean_text("Hello, World! How are you?")
        assert all(c not in ".,!?;:" for c in result)

    def test_non_string_input(self, cleaner):
        """Тест обработки нестрокового ввода."""
        assert cleaner.clean_text(None) == ""
        assert cleaner.clean_text(123) == ""

    def test_empty_string(self, cleaner):
        """Тест пустой строки."""
        assert cleaner.clean_text("") == ""

    def test_clean_texts(self, cleaner):
        """Тест очистки списка текстов."""
        texts = ["Hello World", "Test 123", "<p>HTML</p>"]
        cleaned = cleaner.clean_texts(texts)
        assert len(cleaned) == 3
        assert isinstance(cleaned[0], str)

    def test_lemmatization(self, full_cleaner):
        """Тест лемматизации."""
        # Лемматизация должна работать, но точный результат зависит от NLTK
        result = full_cleaner.clean_text("running dogs")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_call_method(self, cleaner):
        """Тест вызова объекта как функции."""
        assert cleaner("TEST") == cleaner.clean_text("TEST")
