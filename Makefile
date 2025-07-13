.PHONY: help install test lint format security clean build run

help: ## Показать справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	poetry install

install-dev: ## Установить зависимости для разработки
	poetry install --with dev

test: ## Запустить тесты
	poetry run pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-fast: ## Запустить быстрые тесты (без coverage)
	poetry run pytest tests/ -v -m "not slow"

test-unit: ## Запустить только unit тесты
	poetry run pytest tests/ -v -m "unit"

test-integration: ## Запустить только integration тесты
	poetry run pytest tests/ -v -m "integration"

lint: ## Запустить линтеры
	poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	poetry run black --check .
	poetry run isort --check-only .
	poetry run mypy main.py --ignore-missing-imports

format: ## Форматировать код
	poetry run black .
	poetry run isort .

security: ## Проверить безопасность
	poetry run bandit -r . -f json -o bandit-report.json || true
	poetry run safety check --json --output safety-report.json || true
	@echo "Отчёты безопасности сохранены в bandit-report.json и safety-report.json"

clean: ## Очистить временные файлы
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/

build: ## Собрать пакет
	poetry build

run: ## Запустить приложение
	poetry run streamlit run main.py

run-dev: ## Запустить приложение в режиме разработки
	OPENAI_API_KEY="test-key" poetry run streamlit run main.py

ci: ## Запустить все проверки CI
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security
	$(MAKE) build

pre-commit: ## Подготовка к коммиту
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-fast

docker-build: ## Собрать Docker образ
	docker build -t aai:latest .

docker-run: ## Запустить Docker контейнер
	docker run -p 8501:8501 -e OPENAI_API_KEY=your-key-here aai:latest