#!/usr/bin/env bash

uv run ruff check --fix .
uv run ruff check --select I --fix .
uv run --dev mypy --disable-error-code=import-untyped --show-error-context --check-untyped-defs src tests
