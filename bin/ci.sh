#!/usr/bin/env bash
set -euo pipefail

uv run --group dev --locked ruff check src tests
uv run --group dev --locked ruff format --check src tests
uv run --group dev --locked mypy --disable-error-code=import-untyped --show-error-context --check-untyped-defs src tests
uv run --group dev --locked pytest -q \
  tests/test_data_lib.py \
  tests/test_model_cache.py \
  tests/test_nlp_utils.py \
  tests/test_plotting_label_mapping.py \
  tests/test_streamlit_caches.py \
  tests/test_transformers_utils.py \
  tests/test_umap_backend.py
