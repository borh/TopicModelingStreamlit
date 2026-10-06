from typing import Any

import pandas as pd
import streamlit as st

from topic_modeling_streamlit.cache_locks import device_compute_lock
from topic_modeling_streamlit.security import validate_text


@st.cache_data(max_entries=5, ttl=3600, show_spinner=True)
def find_topics(
    cache_key: str, _model: Any, query: str, device: str = "cpu"
) -> tuple[list[int], list[float]]:
    _ = cache_key
    validate_text(query)
    with device_compute_lock(device):
        return _model.find_topics(query)


def visualize_text(
    cache_key: str,
    _model: Any,
    doc: str,
    use_embedding_model: bool,
    language: str,
    device: str = "cpu",
) -> Any:
    distributions = _text_distribution(
        cache_key, _model, doc, use_embedding_model, language, device
    )
    return _model.visualize_approximate_distribution(doc, distributions[0])


@st.cache_data(max_entries=5, ttl=3600, show_spinner=True)
def _text_distribution(
    cache_key: str,
    _model: Any,
    doc: str,
    use_embedding_model: bool,
    language: str,
    device: str,
) -> Any:
    validate_text(doc)
    with device_compute_lock(device):
        _, distributions = _model.approximate_distribution(
            doc,
            use_embedding_model=use_embedding_model,
            calculate_tokens=True,
            min_similarity=0.001,
            separator="" if language == "Japanese" else " ",
        )
    return distributions


@st.cache_data(max_entries=5, ttl=3600, show_spinner=True)
def topics_per_class(
    cache_key: str, _model: Any, docs: list[str], classes: list[str]
) -> pd.DataFrame:
    return _model.topics_per_class(docs, classes)
