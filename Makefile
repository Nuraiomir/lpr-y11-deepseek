SERVICE_NAME := example-service
PORT ?= 8080

.DEFAULT_GOAL := help

.PHONY: help install run test test-models build clean format lint dev

help:  ## Показать команды
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Поставить зависимости
	uv sync --extra dev

run:  ## Запустить сервис локально
	MODEL_PATH=$${MODEL_PATH:-model/best11n_512.onnx} \
	SERVE_DEMO_PAGE=true \
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:  ## Тесты, которым не нужны модели (то же, что гоняет CI)
	uv run pytest tests/ -v

test-models:  ## Тесты с настоящими моделями, на машине где они есть
	uv run pytest tests/ -v -m models

build:  ## Собрать образ
	docker build -t $(SERVICE_NAME):dev .

format:  ## Отформатировать код
	uv run black app tests

lint:  ## Линтер и проверка типов
	uv run ruff check app tests
	uv run mypy app

clean:  ## Убрать сгенерированное
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

dev: install run  ## Поставить и запустить
