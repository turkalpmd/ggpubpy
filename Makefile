.PHONY: help install install-dev test lint format clean build upload docs docs-clean docs-serve

help:
	@echo "Available commands:"
	@echo "  install     Install package for production"
	@echo "  install-dev Install package for development"
	@echo "  test        Run tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  upload      Upload to PyPI"
	@echo "  docs        Build documentation"
	@echo "  docs-clean  Clean documentation build"
	@echo "  docs-serve  Build and serve docs locally"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -v --cov=ggpubpy --cov-report=html --cov-report=term

lint:
	flake8 ggpubpy tests examples
	mypy ggpubpy
	black --check ggpubpy tests examples
	isort --check-only ggpubpy tests examples

format:
	black ggpubpy tests examples
	isort ggpubpy tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	twine check dist/*
	twine upload dist/*

docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html/"

docs-clean:
	rm -rf docs/_build/

docs-serve: docs
	python -m http.server 8000 -d docs/_build/html/
