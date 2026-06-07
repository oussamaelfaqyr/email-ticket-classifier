.PHONY: setup train serve lint clean dvc-init

setup:
	python -m venv .venv
	.venv\Scripts\pip install -e .
	.venv\Scripts\pip install -e .[dev]

train:
	python -m src.pipeline.run_pipeline

dvc-init:
	dvc init
	git commit -m "Initialize DVC"

serve:
	uvicorn src.serving.app:app --reload

lint:
	ruff check .
	mypy .

clean:
	rm -rf data/raw/* data/processed/* models/*
