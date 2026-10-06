import csv
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from topic_modeling_streamlit import cache_locks, public_analyses, security

APP_DIR = Path(__file__).resolve().parents[1] / "src/topic_modeling_streamlit"


@pytest.mark.parametrize(
    "headers,allowed",
    [
        ({}, False),
        ({"X-Topic-Client-IP": "133.1.0.0"}, True),
        ({"X-Topic-Client-IP": "133.1.255.255"}, True),
        ({"X-Topic-Client-IP": "133.2.0.1"}, False),
        ({"X-Topic-Client-IP": "::1"}, False),
        ({"X-Topic-Client-IP": "133.1.2.3, 203.0.113.1"}, False),
        ({"X-Forwarded-For": "133.1.2.3"}, False),
        ({"X-Topic-Client-IP": "203.0.113.1", "X-Forwarded-For": "133.1.2.3"}, False),
    ],
)
def test_public_access_requires_proxy_supplied_university_address(
    monkeypatch, headers, allowed
):
    monkeypatch.setenv("TOPIC_MODELING_PUBLIC", "1")
    monkeypatch.setattr(st, "context", SimpleNamespace(headers=headers))
    assert security.full_access() is allowed


def test_local_development_does_not_require_proxy(monkeypatch):
    monkeypatch.delenv("TOPIC_MODELING_PUBLIC", raising=False)
    assert security.full_access()


@pytest.mark.parametrize("app", ["bertopic", "gensim"])
def test_outside_visitors_see_published_results_without_compute_or_uploads(
    tmp_path, monkeypatch, app
):
    monkeypatch.chdir(tmp_path)
    topics = b"topic_id,label\n0,example\n"
    public_analyses.publish_analysis(app, "a" * 16, topics, {"topics": topics})
    monkeypatch.setenv("TOPIC_MODELING_PUBLIC", "1")
    at = AppTest.from_file(str(APP_DIR / f"{app}_app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.dataframe[0].value.label.tolist() == ["example"]
    assert not at.get("file_uploader")
    assert not at.text_input
    assert not at.text_area
    assert not at.button
    assert len(at.get("download_button")) == 1
    assert list((tmp_path / "cache").glob("**/*")) == []


def test_public_cache_misses_never_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOPIC_MODELING_PUBLIC", "1")
    at = AppTest.from_file(str(APP_DIR / "gensim_app.py"), default_timeout=30).run()
    assert not at.exception
    assert "No analyses" in at.info[-1].value
    assert not at.button
    for lock in (
        cache_locks.model_cache_lock(tmp_path / "model.lock"),
        cache_locks.device_compute_lock("cpu"),
    ):
        with pytest.raises(PermissionError), lock:
            pytest.fail("Public computation must not start")
    assert not (tmp_path / "model.lock").exists()


def test_publishing_is_explicit_bounded_and_atomic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    topics = b"topic_id,label\n0,example\n"
    public_analyses.publish_analysis("gensim", "a" * 16, topics, {"topics": topics})
    path = tmp_path / "published/gensim" / f"{'a' * 16}.json"
    original = path.read_bytes()
    monkeypatch.setattr(public_analyses, "MAX_ANALYSIS_BYTES", 1)
    with pytest.raises(ValueError, match="10 MB"):
        public_analyses.publish_analysis("gensim", "a" * 16, topics, {})
    assert path.read_bytes() == original
    monkeypatch.setattr(public_analyses, "MAX_ANALYSIS_BYTES", 10000)
    monkeypatch.setattr(public_analyses, "MAX_ANALYSES", 1)
    with pytest.raises(ValueError, match="publication limit"):
        public_analyses.publish_analysis("gensim", "b" * 16, topics, {})
    with pytest.raises(ValueError, match="identifier"):
        public_analyses.publish_analysis("gensim", "../private", topics, {})
    monkeypatch.setenv("TOPIC_MODELING_PUBLIC", "1")
    with pytest.raises(PermissionError):
        public_analyses.publish_analysis("gensim", "a" * 16, topics, {})
    assert path.read_bytes() == original


def test_eight_requests_are_admitted_and_ninth_retries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    admitted = Barrier(9)
    release = Event()

    def compute(user):
        with (
            cache_locks.model_cache_lock(tmp_path / f"{user}.lock"),
            cache_locks.model_cache_lock(tmp_path / f"nested-{user}.lock"),
        ):
            admitted.wait(timeout=5)
            assert release.wait(timeout=5)
        return user

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(compute, user) for user in range(8)]
        try:
            admitted.wait(timeout=5)
            with (
                pytest.raises(ValueError, match="eight computation slots"),
                cache_locks.device_compute_lock("cpu"),
            ):
                pytest.fail("Ninth request must not start")
        finally:
            release.set()
        assert [future.result(timeout=5) for future in futures] == list(range(8))
    with cache_locks.device_compute_lock("cpu"):
        pass


def test_input_and_disk_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    security.validate_text("x" * security.MAX_TEXT_CHARS)
    with pytest.raises(ValueError, match="characters"):
        security.validate_text("x" * (security.MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="bytes"):
        security.validate_upload(b"x" * (security.MAX_UPLOAD_BYTES + 1))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/model").write_bytes(b"x" * 10)
    monkeypatch.setattr(security, "MAX_CACHE_BYTES", 10)
    with pytest.raises(ValueError, match="cache is full"):
        security.check_cache_capacity(1)


def test_csv_exports_escape_formulas_without_changing_numeric_values():
    table = pd.DataFrame(
        {
            "=header": [
                '=HYPERLINK("https://example.invalid")',
                " +1",
                "@SUM(1)",
                "normal",
            ],
            "value": [-1, 0, 0.25, 1],
        }
    )
    rows = list(csv.reader(io.StringIO(security.csv_bytes(table).decode())))
    assert rows[0][0] == "'=header"
    assert [row[0] for row in rows[1:]] == [
        '\'=HYPERLINK("https://example.invalid")',
        "' +1",
        "'@SUM(1)",
        "normal",
    ]
    assert [row[1] for row in rows[1:]] == ["-1.0", "0.0", "0.25", "1.0"]
