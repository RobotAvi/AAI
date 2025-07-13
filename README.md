# AAI (Audio Analysis and Intelligence)

Веб-приложение для автоматического анализа и создания отчётов по встречам на основе аудио/видео записей.

## Возможности

- 🎥 **Извлечение аудио из видео** - конвертация MP4 в MP3
- 🎤 **Распознавание речи** - преобразование аудио в текст с помощью OpenAI Whisper
- 📝 **Создание саммари** - анализ текста разговора и создание структурированного отчёта с помощью GPT-4
- 🔄 **Обработка больших файлов** - автоматическое разбиение на чанки
- 🎯 **Специализация на деловых встречах** - оптимизировано для анализа переговоров между сотрудниками

## Технологический стек

- **Python 3.10+** - основной язык программирования
- **Streamlit** - веб-интерфейс
- **OpenAI API** - Whisper для распознавания речи, GPT-4 для анализа
- **Pydub** - обработка аудио файлов
- **Poetry** - управление зависимостями
- **Docker** - контейнеризация

## Быстрый старт

### Предварительные требования

* Python 3.10+
* [Poetry](https://python-poetry.org/)
* ffmpeg
* OpenAI API ключ

### Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd aai
```

2. Установите зависимости:
```bash
poetry install
```

3. Установите ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Скачайте с https://ffmpeg.org/download.html
```

4. Настройте переменные окружения:
```bash
export OPENAI_API_KEY="ваш-ключ-openai-api"
```

### Запуск

```bash
poetry run streamlit run main.py
```

Приложение будет доступно по адресу: http://localhost:8501

## Разработка

### Установка зависимостей для разработки

```bash
poetry install --with dev
```

### Запуск тестов

```bash
# Все тесты
make test

# Быстрые тесты
make test-fast

# Только unit тесты
make test-unit

# Только integration тесты
make test-integration
```

### Линтинг и форматирование

```bash
# Проверка кода
make lint

# Форматирование кода
make format
```

### Проверка безопасности

```bash
make security
```

### Подготовка к коммиту

```bash
make pre-commit
```

## Docker

### Сборка образа

```bash
make docker-build
```

### Запуск контейнера

```bash
make docker-run
```

### Docker Compose

```bash
# Продакшн
docker-compose up -d

# Разработка
docker-compose --profile dev up -d
```

## CI/CD

Проект использует GitHub Actions для автоматизации:

### Workflows

- **CI/CD Pipeline** (`.github/workflows/ci.yml`)
  - Тестирование на Python 3.10, 3.11, 3.12
  - Линтинг (flake8, black, isort, mypy)
  - Покрытие кода тестами
  - Сборка пакета
  - Проверка безопасности

- **Security Scan** (`.github/workflows/security.yml`)
  - Еженедельное сканирование уязвимостей
  - Bandit для анализа кода
  - Safety для проверки зависимостей
  - Trivy для сканирования контейнеров

- **Deploy** (`.github/workflows/deploy.yml`)
  - Автоматическое создание релизов
  - Загрузка артефактов

### Статус

![CI/CD](https://github.com/username/aai/workflows/CI%2FCD%20Pipeline/badge.svg)
![Security](https://github.com/username/aai/workflows/Security%20Scan/badge.svg)

## Структура проекта

```
aai/
├── .github/workflows/    # GitHub Actions
├── tests/               # Тесты
├── main.py             # Основное приложение
├── pyproject.toml      # Конфигурация Poetry
├── Dockerfile          # Docker образ
├── docker-compose.yml  # Docker Compose
├── Makefile           # Команды для разработки
└── README.md          # Документация
```

## API

### Основные функции

- `extract_audio(uploaded_file)` - извлечение аудио из видео
- `split_audio(audio_buffer, number_of_chunks)` - разделение аудио на чанки
- `speech_to_text(path_to_file)` - распознавание речи
- `summarize(conversation, system_prompt, prompt)` - создание саммари

## TODO

- [ ] Реализовать метод `extract_audio` - извлекает аудио из mp4, нужно использовать библиотеку ffmpeg
- [ ] Реализовать метод `speech_to_text` - преобразовывает разговор в текст, нужно использовать OpenAI, модель Whisper
- [ ] Реализовать метод `summarize` - саммаризирует текст с помощью OpenAI и GPT-3.5/GPT-4
- [ ] (?) Реализовать `speech_to_text` с помощью локальной модели (Whisper)
- [ ] (?) Реализовать `summarize` с помощью локальной модели (Saiga 2 70B, Zephyr 7B или Mistral 7B)
- [ ] Поэкспериментировать с промптами
- [x] Добавить тесты
- [x] Настроить CI/CD
- [x] Добавить Docker поддержку
- [x] Настроить линтеры и форматирование

## Лицензия

MIT License

## Автор

Anton Belousov <anton@belousov.co>
