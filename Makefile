.PHONY: install test lint demo up down replay materialize backfill

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	mypy src

up:
	docker compose up -d --build

down:
	docker compose down

replay:
	python -m energy_grid.cli replay --fixture data/fixtures/events.jsonl

materialize:
	python -m energy_grid.cli materialize-demo

backfill:
	python -m energy_grid.cli backfill --fixture data/fixtures/events.jsonl

demo:
	docker compose up -d --build
	docker compose run --rm demo

