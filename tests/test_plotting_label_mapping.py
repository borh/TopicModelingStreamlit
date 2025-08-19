import numpy as np
import pandas as pd

from topic_modeling_streamlit.plotting import (
    build_topic_label_map,
    visualize_documents_altair,
)


class FakeModel:
    def __init__(self, info_records, topic_terms: dict[int, list] | None = None):
        self._info = pd.DataFrame(info_records)
        self.topic_aspects_ = {}
        self._terms = topic_terms or {}

    def get_topic_info(self):
        return self._info

    def get_topic(self, tid):
        return self._terms.get(int(tid), [])


def test_build_topic_label_map_nonsequential_ids():
    info = [
        {"Topic": 7, "Name": "7: Seven", "CustomName": "7: SevenCustom"},
        {"Topic": 3, "Name": "3: Three", "CustomName": None},
        {"Topic": -1, "Name": "-1: Outlier", "CustomName": None},
    ]
    m = FakeModel(info)
    mp = build_topic_label_map(m)
    assert set(mp.keys()) == {7, 3, -1}
    # Leading "7: " prefix should be stripped
    assert mp[7] == "SevenCustom" or mp[7] == "Seven"
    assert mp[3] in {"Three", "3: Three", "3"} or isinstance(mp[3], str)
    assert mp[-1] in {"Outlier", "-1: Outlier", "-1"}


def test_visualize_documents_respects_passed_topics_and_label_map():
    # Three documents with two distinct non-sequential topic ids: 10 and 2
    docs = ["doc0", "doc1", "doc2"]
    topics_passed = [10, 2, 10]
    # Simple 2D reduced embeddings matching docs order
    reduced_embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    info = [
        {"Topic": 10, "Name": "10: Ten", "CustomName": None},
        {"Topic": 2, "Name": "2: Two", "CustomName": None},
    ]
    fake = FakeModel(info, topic_terms={10: [("ten", 1.0)], 2: [("two", 1.0)]})
    label_map = {10: "Ten", 2: "Two"}
    # Provide chunk ids so we can sort the returned dataframe deterministically
    chunk_ids = [0, 1, 2]

    chart, df = visualize_documents_altair(
        docs=docs,
        topic_model=fake,
        reduced_embeddings=reduced_embeddings,
        sample=1.0,
        custom_labels=False,
        authors=["a", "b", "c"],
        works=["w0", "w1", "w2"],
        chunk_ids=chunk_ids,
        color_by="Topic",
        selectable=False,
        show_outliers=True,
        topics_per_doc=topics_passed,
        label_map=label_map,
        debug_return_df=True,
    )

    # df may be sampled in arbitrary order; sort by chunk_id to recover original ordering
    df_sorted = df.sort("chunk_id")
    assert df_sorted.get_column("chunk_id").to_list() == chunk_ids
    # topics should match the passed topics for each chunk id
    assert df_sorted.get_column("topic").to_list() == topics_passed
    # labels should map via provided label_map
    labels = df_sorted.get_column("label").to_list()
    assert labels[0] == "Ten"
    assert labels[1] == "Two"
    assert labels[2] == "Ten"
