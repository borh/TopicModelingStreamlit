import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pandas as pd
import pytest

from topic_modeling_streamlit.cache_locks import (
    device_compute_lock,
    model_cache_lock,
)
from topic_modeling_streamlit.streamlit_caches import (
    find_topics,
    topics_per_class,
    visualize_text,
)


class TopicSearchModel:
    def __init__(self, result: tuple[list[int], list[float]]) -> None:
        self.result = result
        self.calls = 0

    def find_topics(self, query: str) -> tuple[list[int], list[float]]:
        self.calls += 1
        return self.result


def test_topic_search_cache_isolated_by_model_key() -> None:
    find_topics.clear()
    first_model = TopicSearchModel(([1], [0.9]))
    second_model = TopicSearchModel(([2], [0.8]))

    assert find_topics("model-one", first_model, "shared query") == ([1], [0.9])
    assert find_topics("model-one", first_model, "shared query") == ([1], [0.9])
    assert find_topics("model-two", second_model, "shared query") == ([2], [0.8])

    assert first_model.calls == 1
    assert second_model.calls == 1


@pytest.mark.parametrize("language,separator", [("English", " "), ("Japanese", "")])
def test_text_inference_preserves_language_boundaries(language, separator):
    class Model:
        def approximate_distribution(self, doc, **kwargs):
            return None, [[kwargs["separator"].join(["one", "two"])]]

        def visualize_approximate_distribution(self, doc, distribution):
            return distribution

    assert visualize_text("model", Model(), "one two", False, language) == [
        f"one{separator}two"
    ]


def test_styled_inference_results_do_not_need_to_be_picklable():
    class Model:
        def approximate_distribution(self, doc, **kwargs):
            return None, [[[1.0]]]

        def visualize_approximate_distribution(self, doc, distribution):
            return pd.DataFrame(distribution).style.apply(lambda row: ["color: red"])

    result = visualize_text("styled-model", Model(), "text", False, "English")
    assert result.data.iloc[0, 0] == 1.0


def test_genre_analysis_uses_current_documents_and_metadata():
    class Model:
        def topics_per_class(self, docs, classes):
            return list(zip(docs, classes, strict=True))

    topics_per_class.clear()
    assert topics_per_class("model", Model(), ["a", "b"], ["fiction", "poetry"]) == [
        ("a", "fiction"),
        ("b", "poetry"),
    ]
    assert topics_per_class("model", Model(), ["a"], ["new genre"]) == [
        ("a", "new genre")
    ]


def test_old_model_lock_file_does_not_allow_a_second_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "model.lock"
    first_lock = model_cache_lock(lock_path)
    acquired = Event()

    def acquire_after_first_writer() -> None:
        with model_cache_lock(lock_path):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with first_lock:
            os.utime(lock_path, (0, 0))
            waiting_writer = pool.submit(acquire_after_first_writer)
            assert not acquired.wait(timeout=0.1)
            assert lock_path.exists()

        assert acquired.wait(timeout=2)
        waiting_writer.result(timeout=2)


def test_cuda_and_bitsandbytes_share_the_device_compute_lock(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    acquired = Event()

    def acquire_bitsandbytes_device() -> None:
        with device_compute_lock("cuda:0-bnb"):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with device_compute_lock("cuda:0"):
            waiting_writer = pool.submit(acquire_bitsandbytes_device)
            assert not acquired.wait(timeout=0.1)

        assert acquired.wait(timeout=2)
        waiting_writer.result(timeout=2)


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
def test_five_users_can_reenter_the_device_lock(tmp_path, monkeypatch, device):
    monkeypatch.chdir(tmp_path)
    ready = Barrier(5)
    state = {"active": 0, "peak": 0, "completed": 0}

    def compute():
        ready.wait(timeout=5)
        with device_compute_lock(device):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            with device_compute_lock(device):
                state["completed"] += 1
            state["active"] -= 1

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(compute) for _ in range(5)]
        for future in futures:
            future.result(timeout=10)

    assert state == {"active": 0, "peak": 1, "completed": 5}
