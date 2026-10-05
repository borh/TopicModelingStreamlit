from types import SimpleNamespace

import numpy as np
import pytest
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from topic_modeling_streamlit.gensim_lib import (
    colorize_topics,
    create_dtm,
    infer_text,
)


@pytest.fixture
def trained_lda():
    docs = [["cat", "kitten", "cat"], ["dog", "puppy", "dog"], ["cat", "dog"]]
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    model = LdaModel(
        corpus, id2word=dictionary, num_topics=3, passes=2, random_state=42
    )
    return model, dictionary, corpus


def test_document_probabilities_include_small_topics_and_group_means(trained_lda):
    model, _, corpus = trained_lda
    model.minimum_probability = 0.8
    docs = create_dtm("dense", model, 3, corpus, ["A", "A", "B"], collapsed=False)
    np.testing.assert_allclose(docs.sum(axis=1), 1, atol=1e-6)
    assert docs.shape == (3, 3)
    model.random_state.seed(42)
    grouped = create_dtm("grouped", model, 3, corpus, ["A", "A", "B"])
    model.random_state.seed(42)
    expected = create_dtm(
        "grouped-expected", model, 3, corpus, ["A", "A", "B"], collapsed=False
    )
    np.testing.assert_allclose(grouped, expected.groupby(level=0).mean(), atol=1e-6)


def test_inference_uses_training_preprocessing_and_preserves_surface_text():
    tokens = [
        SimpleNamespace(
            surface="猫たち", feature=SimpleNamespace(lemma="猫", pos2="一般")
        ),
        SimpleNamespace(
            surface="東京", feature=SimpleNamespace(lemma="東京", pos2="固有名詞")
        ),
    ]
    dictionary = Dictionary([["猫", "東京"]])
    processed, surfaces, bow = infer_text("ignored", lambda _: tokens, dictionary)
    assert processed == ["猫", ""]
    assert surfaces == ["猫たち", "東京"]
    assert bow == [(dictionary.token2id["猫"], 1)]


def test_highlighting_escapes_input_and_displays_topic_zero(trained_lda):
    model, dictionary, corpus = trained_lda
    html = colorize_topics(
        0, dictionary, model, corpus, [["cat"]], [["<script>"]], ["<title>"]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<title>" not in html


def test_topic_zero_annotation():
    from topic_modeling_streamlit.gensim_lib import colorize

    assert "<rt>0</rt>" in colorize("cat", "black", topicid=0)


@pytest.mark.parametrize("limit,expected", [(0, [2, 2]), (1, [1, 1])])
def test_chunk_controls_keep_tail_surfaces_and_metadata_aligned(
    tmp_path, monkeypatch, limit, expected
):
    import polars as pl

    from topic_modeling_streamlit import gensim_lib
    from topic_modeling_streamlit.data_lib import create_chunked_data

    def tagger(text):
        return [
            SimpleNamespace(
                surface=word, feature=SimpleNamespace(lemma=word.lower(), pos2="一般")
            )
            for word in text.split()
        ]

    monkeypatch.setattr(gensim_lib, "get_tagger", lambda: tagger)
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "Aozora-Bunko-Fiction-Selection-2022-05-30" / "Plain"
    directory.mkdir(parents=True)
    (directory / "b.txt").write_text("B B B。\nTail B", encoding="utf-8")
    (directory / "a.txt").write_text("A A A。\nTail A", encoding="utf-8")
    metadata = pl.DataFrame(
        {
            "filename": ["a.txt", "b.txt"],
            "author_ja": ["Author A", "Author B"],
            "title_ja": ["Work A", "Work B"],
            "genre": ["A", "B"],
            "year": [2000, 2001],
        }
    )
    enriched, docs, surfaces = create_chunked_data(
        metadata, chunksize=3, min_chunksize=1, chunks=limit
    )
    assert (
        enriched.group_by("filename").len().sort("filename")["len"].to_list()
        == expected
    )
    assert enriched["docid"].to_list() == list(range(len(docs)))
    for row, tokens, original in zip(enriched.iter_rows(named=True), docs, surfaces):
        assert tokens == [word.lower() for word in original]
        assert row["title"][-1].lower() in "".join(tokens)
    filtered, filtered_docs, _ = create_chunked_data(
        metadata, chunksize=3, min_chunksize=3
    )
    assert len(filtered_docs) == filtered.height == 2
