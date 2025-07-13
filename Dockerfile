# Используем официальный Python образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN pip install poetry

# Копируем файлы конфигурации Poetry
COPY pyproject.toml poetry.lock* ./

# Настраиваем Poetry для работы в контейнере
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Устанавливаем зависимости
RUN poetry install --only=main --no-root

# Копируем исходный код
COPY . .

# Устанавливаем приложение
RUN poetry install --only=main

# Создаем пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Открываем порт
EXPOSE 8501

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Запускаем приложение
CMD ["poetry", "run", "streamlit", "run", "main.py"]