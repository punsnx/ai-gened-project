PYTHON ?= python
PIP ?= pip
IMAGE ?= ai-gened-backend
PORT ?= 8000

.PHONY: install dev build run

install:
	$(PIP) install -r requirements-dev.txt

dev:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

build:
	docker build -t $(IMAGE):latest .

run:
	docker run --rm -p $(PORT):8000 $(IMAGE):latest
