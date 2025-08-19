import numpy as np
import polars as pl

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
