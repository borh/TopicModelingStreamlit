from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from topic_modeling_streamlit import gensim_lib

APP = Path(__file__).resolve().parents[1] / "src/topic_modeling_streamlit/gensim_app.py"


@pytest.fixture
def lda_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "Aozora-Bunko-Fiction-Selection-2022-05-30"
    (root / "Plain").mkdir(parents=True)
    rows = []
    for i in range(4):
        word = "cat" if i < 2 else "dog"
        (root / "Plain" / f"book-{i}.txt").write_text(
            (f"{word} pet animal home family garden。\n" * 4), encoding="utf-8"
        )
        rows.append(
            {
                "filename": f"book-{i}.txt",
                "author_ja": f"Author {i // 2}",
                "title_ja": f"Book {i}",
                "genre": word,
                "year": 2000,
            }
        )
    pd.DataFrame(rows[::-1]).to_csv(root / "groups.csv", sep="\t", index=False)

    def tagger(text):
        return [
            SimpleNamespace(
                surface=word, feature=SimpleNamespace(lemma=word.lower(), pos2="一般")
            )
            for word in text.split()
        ]

    monkeypatch.setattr(gensim_lib, "get_tagger", lambda: tagger)
    return AppTest.from_file(str(APP), default_timeout=30).run()


def number(app, label):
    return next(widget for widget in app.number_input if widget.label == label)


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def train(app):
    for label, value in [
        ("Topics", 3),
        ("Passes", 1),
        ("Iterations", 10),
        ("Chunk size (tokens)", 10),
        ("Min chunk size (tokens)", 1),
        ("Min document frequency", 1),
        ("Max document frequency", 1.0),
    ]:
        number(app, label).set_value(value)
    return button(app, "Compute!").click().run()


def test_training_is_explicit_and_labels_survive_navigation(lda_app):
    assert not lda_app.exception
    assert "lda_result" not in lda_app.session_state
    app = train(lda_app)
    assert not app.exception
    key, model, _, corpus, metadata, _, originals = app.session_state["lda_result"]
    assert model.num_topics == 3
    assert metadata["docid"].to_list() == list(range(len(corpus)))
    np.testing.assert_allclose(
        app.dataframe[0].value.groupby("topic_id").probability.sum(), 1, atol=1e-5
    )
    assert len(app.text) == 5
    probabilities = gensim_lib.create_dtm(
        key, model, 3, corpus, metadata["author"].to_list(), collapsed=False
    ).reset_index(drop=True)
    expected = probabilities[0].nlargest(5).index.tolist()
    assert [text.value for text in app.text] == [
        "".join(originals[i]) for i in expected
    ]
    number(app, "Topics").set_value(4).run()
    assert app.session_state["lda_result"][1].num_topics == 3
    app.text_input[0].set_value("Pets")
    button(app, "Save label").click().run()
    assert app.session_state["lda_labels"] == {0: "Pets"}
    assert app.dataframe[0].value.query("topic_id == 0").label.eq("Pets").all()
    app.selectbox[0].select(1).run()
    assert app.session_state["lda_labels"] == {0: "Pets"}
    assert len(app.get("download_button")) == 5
    assert all(download.proto.ignore_rerun for download in app.get("download_button"))
    app.text_area[0].set_value("CAT CAT CAT")
    button(app, "Infer topics").click().run()
    assert not app.exception
    inference = app.dataframe[-1].value
    assert list(inference.columns) == ["Topic", "Probability"]
    assert inference.Probability.sum() == pytest.approx(1, abs=1e-6)
    app.text_area[0].set_value("unknownword")
    button(app, "Infer topics").click().run()
    assert any("No vocabulary matches" in warning.value for warning in app.warning)
    assert not app.exception


def test_failed_training_preserves_previous_model_and_sessions_are_isolated(lda_app):
    first = train(lda_app)
    first.text_input[0].set_value("Private label")
    button(first, "Save label").click().run()
    second = train(AppTest.from_file(str(APP), default_timeout=30).run())
    assert second.session_state["lda_labels"] == {}
    assert (
        second.session_state["lda_result"][1]
        is not first.session_state["lda_result"][1]
    )
    old_key = first.session_state["lda_result"][0]
    number(first, "Min document frequency").set_value(100)
    button(first, "Compute!").click().run()
    assert not first.exception
    assert any("No vocabulary remains" in error.value for error in first.error)
    assert first.session_state["lda_result"][0] == old_key
    assert first.session_state["lda_labels"][0] == "Private label"
