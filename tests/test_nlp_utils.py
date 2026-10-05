import pickle

import numpy as np
import polars as pl

from topic_modeling_streamlit import nlp_utils
from topic_modeling_streamlit.nlp_utils import stratified_weighted_document_sampling


def test_stratified_weighted_sampling_simple():
    docs = [f"d{i}" for i in range(6)]
    # topics 0 and 1 each have 3 docs
    topics = [0, 0, 0, 1, 1, 1]
    # assign probabilities such that index 0,1 heavily favor topic 0, 2 strongly favors topic 1
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    # build metadata with two strata: "A" and "B"
    metadata = pl.DataFrame({"genre": ["A", "A", "B", "A", "B", "B"]})
    # sample 2 docs per topic
    out = stratified_weighted_document_sampling(
        docs, metadata, topics, probs, nr_docs=2, stratify_by="genre"
    )
    # must have keys 0 and 1
    assert set(out.keys()) == {0, 1}
    # each list has exactly 2 docs
    assert all(len(v) == 2 for v in out.values())
    # for topic 0, one doc from stratum "A" and one from "B"
    sel0 = out[0]
    assert any(d in ["d0", "d1"] for d in sel0)
    assert "d2" in sel0


def test_stratify_missing_column():
    docs = ["x", "y", "z"]
    topics = [0, 0, 0]
    probs = np.zeros((3, 1))
    # metadata without the column → all in one stratum
    metadata = pl.DataFrame({"foo": [1, 2, 3]})
    out = stratified_weighted_document_sampling(
        docs, metadata, topics, probs, nr_docs=3, stratify_by="genre"
    )
    # only topic 0 present
    assert set(out.keys()) == {0}
    # should return all docs (order may vary)
    assert set(out[0]) == set(docs)


def test_sampling_fills_request_when_only_one_document_has_positive_weight():
    docs = ["a", "b", "c", "d"]
    sampled = stratified_weighted_document_sampling(
        docs,
        pl.DataFrame({"genre": ["fiction"] * 4}),
        [0] * 4,
        np.array([[1.0], [0.0], [0.0], [0.0]]),
        nr_docs=3,
        seed=42,
    )
    assert len(sampled[0]) == 3
    assert "a" in sampled[0]


def test_language_processor_pickle_preserves_surface_filter(monkeypatch):
    class Tokenizer(nlp_utils.SpacyTokenizer):
        def __init__(self, *args):
            pass

        def tokenize(self, text):
            return text.split()

    monkeypatch.setitem(nlp_utils.TOKENIZER_MAP, "spaCy", Tokenizer)
    processor = nlp_utils.LanguageProcessor(
        language="English",
        tokenizer_type="spaCy",
        dictionary_type="en_core_web_sm",
        surface_filter="^drop$",
    )
    restored = pickle.loads(pickle.dumps(processor))

    assert restored.tokenizer.tokenize("keep drop") == ["keep"]
    assert restored.vectorizer.get_params()["tokenizer"]("keep drop") == ["keep"]


def test_training_results_and_plot_use_the_same_embeddings(monkeypatch):
    from types import SimpleNamespace

    embeddings = np.ones((3, 4))
    encoding_calls = []

    class Embeddings:
        def encode(self, docs, **kwargs):
            encoding_calls.append(kwargs)
            return embeddings

    class Model:
        def __init__(self, **kwargs):
            self.topics_ = [0, 1, 0]
            self.probabilities_ = np.ones((3, 2))

        def fit(self, docs):
            return self

        def transform(self, docs):
            return [1, 1, 1], np.zeros((3, 2))

        def fit_transform(self, docs, embeddings):
            assert embeddings is embedding_array
            return self.topics_, self.probabilities_

    embedding_array = embeddings
    monkeypatch.setattr(nlp_utils, "load_embedding_model", lambda *args: Embeddings())
    monkeypatch.setattr(nlp_utils, "BERTopic", Model)
    monkeypatch.setattr(
        nlp_utils, "validate_vectorizer", lambda vectorizer, language: vectorizer
    )
    monkeypatch.setattr(
        nlp_utils,
        "get_umap_model",
        lambda device: SimpleNamespace(fit_transform=lambda values: values[:, :2]),
    )
    model, actual_embeddings, _, topics, probs = nlp_utils.calculate_model(
        ["a", "b", "c"], object(), "ruri-test", [], None, 0, "cpu", "Japanese"
    )

    assert topics == model.topics_
    np.testing.assert_array_equal(probs, model.probabilities_)
    assert actual_embeddings is embeddings
    assert encoding_calls == [{"show_progress_bar": True, "prompt": "トピック: "}]


def test_dictionary_file_changes_invalidate_tokenizer_revision(tmp_path, monkeypatch):
    dictionary = tmp_path / "dictionary"
    dictionary.mkdir()
    binary = dictionary / "sys.dic"
    binary.write_bytes(b"first dictionary")
    monkeypatch.setitem(
        nlp_utils.DICTIONARY_MAP,
        "test-dictionary",
        nlp_utils.FugashiDictionary(f"-d {dictionary} -r {dictionary}/dicrc"),
    )
    first = nlp_utils.tokenizer_revision("MeCab", "test-dictionary")
    binary.write_bytes(b"updated dictionary with more entries")
    assert nlp_utils.tokenizer_revision("MeCab", "test-dictionary") != first


def test_japanese_spacy_revision_tracks_sudachi_core(monkeypatch):
    versions = {
        "spacy": "3.8",
        "ja_core_news_sm": "3.8",
        "sudachipy": "0.7",
        "sudachidict-core": "old",
    }
    monkeypatch.setattr(nlp_utils, "version", versions.__getitem__)
    first = nlp_utils.tokenizer_revision("spaCy", "ja_core_news_sm")
    versions["sudachidict-core"] = "updated"
    assert nlp_utils.tokenizer_revision("spaCy", "ja_core_news_sm") != first
