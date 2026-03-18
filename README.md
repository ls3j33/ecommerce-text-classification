# E-commerce Classification ML Service

ML-сервис для классификации товаров электронной коммерции по категориям.

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установка uv (если не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Клонирование репозитория
git clone <repository-url>
cd ecommerce-classification

# Создание виртуального окружения и установка зависимостей
uv venv
uv sync
```

### 2. Настройка конфигурации

```bash
# Копирование шаблона
cp .env.example .env

# Отредактируйте .env при необходимости
```

### 3. Запуск через Docker Compose

```bash
# Inference сервис (24/7)
docker compose up -d inference

# Training сервис (по требованию)
docker compose --profile train up training
```

### 4. Локальный запуск

```bash
# Обучение модели (SVM)
python -m src.training.main

# Inference сервер
uvicorn src.inference.main:app --reload
```

---

## 📡 API Endpoints

### Health Check
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### Предсказание
```powershell
$body = @{description = "Sony wireless headphones"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body $body
```

### Смена модели
```powershell
$body = @{version = "v1"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/change-model" -Method POST -ContentType "application/json" -Body $body
```

### Swagger UI
Откройте в браузере: **http://127.0.0.1:8000/docs**

---

## 🏗️ Архитектура

```
ecommerce-classification/
├── src/
│   ├── preprocessing/     # Общий препроцессинг текста
│   ├── inference/         # FastAPI сервис (24/7)
│   └── training/          # Обучение моделей (по запросу)
├── models/registry/       # Реестр версий моделей
├── tests/                 # Юнит-тесты
├── docker-compose.yml     # Оркестрация
└── pyproject.toml         # Зависимости
```

### Inference Сервис
- **FastAPI** приложение
- REST API для предсказаний
- Health check endpoint
- Смена версии модели

### Training Сервис
- Запускается по требованию
- Обучает новую модель
- Валидирует метрики (F1 >= 0.95)
- Регистрирует в реестре
- Автоматически останавливается

---

## 📊 Поддерживаемые модели

| Модель | Время обучения | F1-macro |
|--------|----------------|----------|
| Linear SVM | ~2 мин | 0.95+ |
| Logistic Regression | ~2 мин | 0.95+ |
| DistilBERT | ~10 мин | 0.96+ |

---

## 🧪 Тестирование

```bash
# Запуск тестов
uv run pytest tests/ -v

# С покрытием
uv run pytest tests/ --cov=src --cov-report=html
```

---

## 🔧 Конфигурация

Параметры в `.env`:

```env
# Model
MODEL_TYPE=svm              # svm, distilbert, logistic
MIN_F1_THRESHOLD=0.95       # Минимальный порог F1

# Inference
INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000

# Training
TRAIN_BATCH_SIZE=32
TRAIN_EPOCHS=4
TRAIN_LEARNING_RATE=5e-5
```

---

## 📁 Структура проекта

```
ecommerce-classification/
├── data/
│   ├── raw/                      # Исходные данные (не в git)
│   └── processed/                # Обработанные данные (не в git)
├── models/
│   └── registry/                 # Реестр моделей (не в git)
├── src/
│   ├── config.py                 # Конфигурация
│   ├── logging_config.py         # Логирование
│   ├── preprocessing/
│   │   └── text_cleaner.py       # Очистка текста
│   ├── inference/
│   │   ├── main.py               # FastAPI приложение
│   │   ├── models.py             # Pydantic модели
│   │   └── model_loader.py       # Загрузка моделей
│   └── training/
│       ├── main.py               # Скрипт обучения
│       ├── train.py              # Логика тренировки
│       ├── validate.py           # Валидация метрик
│       ├── deploy.py             # Деплой в реестр
│       └── model_registry.py     # Реестр моделей
├── tests/
│   ├── test_preprocessing.py
│   ├── test_inference.py
│   └── test_training.py
├── .env.example
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile.inference
├── Dockerfile.training
└── README.md
```

---

## 🔄 Workflow

```
1. docker compose up -d inference     # Запуск inference
                                      ↓
2. docker compose --profile train up training  # Обучение
                                      ↓
3. Training:
   - Загружает данные
   - Обучает модель
   - Валидирует метрики
   - Регистрирует в реестре
                                      ↓
4. Inference подгружает новую модель
```

---

## 📦 Зависимости

Основные:
- `fastapi` - REST API
- `uvicorn` - ASGI сервер
- `pydantic` - Валидация данных
- `scikit-learn` - ML модели
- `pandas` - Обработка данных
- `nltk` - NLP
- `pytest` - Тестирование

Полный список в `pyproject.toml`.

---

## 📝 License

MIT
