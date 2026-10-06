import builtins
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

from topic_modeling_streamlit import nlp_utils


@pytest.fixture
def model_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    processors = []

    class Processor:
        def __init__(self, **kwargs):
            processors.append(kwargs)
            self.vectorizer = SimpleNamespace(vocabulary_={"kept": 0})

    class Model:
        def __init__(self):
            self.vectorizer_model = Processor().vectorizer
            self.topics_ = [0, 0]
            self.probabilities_ = np.ones((2, 1))
            self.topic_aspects_ = {"Main": {0: [("kept", 1.0)]}}
            self.custom_labels_ = ["0: kept"]

        def save(self, path, **kwargs):
            assert kwargs["save_embedding_model"] is False
            Path(path).mkdir(exist_ok=True)

        def set_topic_labels(self, labels):
            self.custom_labels_ = labels

    computations = []

    def compute(*args, **kwargs):
        computations.append(args)
        model = Model()
        return (
            model,
            np.ones((2, 3)),
            np.ones((2, 2)),
            model.topics_,
            model.probabilities_,
        )

    monkeypatch.setattr(nlp_utils, "LanguageProcessor", Processor)
    monkeypatch.setattr(nlp_utils, "calculate_model", compute)
    monkeypatch.setattr(nlp_utils, "load_embedding_model", lambda *args: object())
    monkeypatch.setattr(nlp_utils.BERTopic, "load", lambda *args, **kwargs: Model())

    def load(**kwargs):
        settings = {
            "docs": ["first document", "second document"],
            "language": "English",
            "embedding_model": "cl-nagoya/ruri-v3-30m",
            "representation_model": ["KeyBERTInspired"],
            "prompt": None,
            "nr_topics": 0,
            "tokenizer_type": "spaCy",
            "dictionary_type": "en_core_web_sm",
            "ngram_range": (1, 1),
            "max_df": 0.9,
            "tokenizer_features": ["orth"],
            "tokenizer_pos_filter": None,
            "surface_filter": "drop",
        }
        settings.update(kwargs)
        return nlp_utils.load_and_persist_model(**settings)

    return load, processors, computations


def test_model_cache_round_trip_preserves_tokenizer_and_arrays(model_cache):
    load, processors, computations = model_cache
    first = load()
    second = load()

    assert first[0] == second[0]
    assert len(computations) == 1
    assert processors[-1]["surface_filter"] == "drop"
    assert second[1].custom_labels_ == ["0: kept"]
    assert second[1].topic_aspects_ == first[1].topic_aspects_
    for before, after in zip(first[2:], second[2:], strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("sidecar", ["topics", "topic_aspects", "custom_labels"])
def test_failed_save_does_not_leave_a_completed_cache(
    model_cache, monkeypatch, sidecar
):
    load, _, _ = model_cache
    path, model, embeddings, reduced_embeddings, _, _ = load()

    original_open = builtins.open

    def fail(path, mode="r", *args, **kwargs):
        if mode == "wb" and str(path).endswith(f"-{sidecar}.pickle"):
            raise OSError("disk full")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail)
    with pytest.raises(OSError, match="disk full"):
        load(
            topic_model=model,
            embeddings=embeddings,
            reduced_embeddings=reduced_embeddings,
        )

    assert not path.with_suffix(".finished").exists()


def test_cache_distinguishes_representation_order(model_cache):
    load, _, computations = model_cache
    first = load(representation_model=["KeyBERTInspired", "MaximalMarginalRelevance"])
    second = load(representation_model=["MaximalMarginalRelevance", "KeyBERTInspired"])

    assert first[0] != second[0]
    assert len(computations) == 2


def test_eight_sessions_share_computation_and_keep_labels_separate(model_cache):
    load, _, computations = model_cache
    ready = Barrier(8)

    def session(user):
        ready.wait(timeout=5)
        result = load()
        result[1].custom_labels_[0] = f"user {user}"
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(session, range(8)))

    assert len(computations) == 1
    assert len({result[0] for result in results}) == 1
    assert [result[1].custom_labels_ for result in results] == [
        [f"user {user}"] for user in range(8)
    ]


def test_dictionary_revision_invalidates_persisted_model(model_cache, monkeypatch):
    load, _, computations = model_cache
    monkeypatch.setattr(nlp_utils, "tokenizer_revision", lambda *args: ("first",))
    first = load()
    monkeypatch.setattr(nlp_utils, "tokenizer_revision", lambda *args: ("updated",))
    second = load()
    assert first[0] != second[0]
    assert len(computations) == 2
