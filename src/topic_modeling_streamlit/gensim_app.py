import os
import socket
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import xxhash
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from topic_modeling_streamlit.cache_locks import device_compute_lock
from topic_modeling_streamlit.data_lib import create_chunked_data, get_metadata
from topic_modeling_streamlit.gensim_lib import (
    colorize_topics,
    create_dtm,
    create_dtm_heatmap,
    get_tagger,
    infer_text,
    pyldavis_html,
)

st.set_page_config(layout="wide")
st.title("Gensim (LDA) を使用したトピックモデル")

with st.sidebar.form("lda_settings"):
    st.subheader("LDA settings")
    random_state = st.number_input("Random state", min_value=0, value=42)
    num_topics = st.number_input("Topics", min_value=2, value=40)
    iterations = st.number_input("Iterations", min_value=1, value=2000)
    chunksize = st.number_input("Training batch size", min_value=10, value=4000)
    passes = st.number_input("Passes", min_value=1, value=15)
    st.subheader("Corpus settings")
    chunk_size = st.number_input("Chunk size (tokens)", min_value=10, value=2000)
    min_chunk_size = st.number_input("Min chunk size (tokens)", min_value=1, value=20)
    chunks_per_work = st.number_input("Chunks per work (0 = all)", min_value=0, value=0)
    min_df = st.number_input("Min document frequency", min_value=1, value=5)
    max_df = st.number_input(
        "Max document frequency", min_value=0.01, max_value=1.0, value=0.5
    )
    compute = st.form_submit_button("Compute!")
st.sidebar.caption(f"Running on {socket.gethostname()}")
with st.sidebar.expander("LDA model"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4d/Smoothed_LDA.png")
    st.markdown(
        "$\\theta$ is each document's topic distribution; $\\varphi$ is each topic's word distribution."
    )


def source_revision() -> tuple[tuple[str, int, int], ...]:
    corpus_dir = Path("Aozora-Bunko-Fiction-Selection-2022-05-30")
    source_files = [
        corpus_dir / "groups.csv",
        *sorted((corpus_dir / "Plain").glob("*.txt")),
    ]
    dictionary_dir = os.environ.get("MECAB_DICDIR_NOVEL")
    if dictionary_dir:
        source_files.extend(
            Path(dictionary_dir) / name
            for name in ("sys.dic", "unk.dic", "matrix.bin", "char.bin", "dicrc")
        )
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_files
        if path.exists()
    )


@st.cache_data(max_entries=5, show_spinner="Preparing corpus…")
def create_cached_chunked_data(
    revision, tokenizer_version, chunk_size, min_chunk_size, chunks_per_work
):
    with device_compute_lock("cpu"):
        return create_chunked_data(
            get_metadata(), chunk_size, min_chunk_size, chunks_per_work
        )


@st.cache_data(max_entries=5, show_spinner="Training LDA…")
def create_lda_model(
    cache_key,
    _docs,
    min_df,
    max_df,
    random_state,
    num_topics,
    chunksize,
    passes,
    iterations,
):
    dictionary = Dictionary([[token for token in doc if token] for doc in _docs])
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)
    if not dictionary:
        raise ValueError(
            "No vocabulary remains. Lower min document frequency or raise max document frequency."
        )
    corpus = [dictionary.doc2bow(doc) for doc in _docs]
    with device_compute_lock("cpu"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            chunksize=chunksize,
            alpha="auto",
            eta="auto",
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            eval_every=None,
        )
    return model, dictionary, corpus


if compute:
    try:
        if min_chunk_size > chunk_size:
            raise ValueError("Min chunk size must not exceed chunk size.")
        revision = source_revision()
        tokenizer_version = version("fugashi-plus")
        corpus_key = xxhash.xxh3_64_hexdigest(
            repr(
                (
                    revision,
                    tokenizer_version,
                    chunk_size,
                    min_chunk_size,
                    chunks_per_work,
                )
            ).encode()
        )
        analysis_key = xxhash.xxh3_64_hexdigest(
            repr(
                (
                    corpus_key,
                    min_df,
                    max_df,
                    random_state,
                    num_topics,
                    chunksize,
                    passes,
                    iterations,
                )
            ).encode()
        )
        metadata, docs, original_docs = create_cached_chunked_data(
            revision, tokenizer_version, chunk_size, min_chunk_size, chunks_per_work
        )
        if not docs:
            raise ValueError(
                "No document chunks were produced. Check the corpus and chunk settings."
            )
        model, dictionary, corpus = create_lda_model(
            analysis_key,
            docs,
            min_df,
            max_df,
            random_state,
            num_topics,
            chunksize,
            passes,
            iterations,
        )
        st.session_state["lda_result"] = (
            analysis_key,
            model,
            dictionary,
            corpus,
            metadata,
            docs,
            original_docs,
        )
        st.session_state["lda_labels"] = {}
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))

if "lda_result" not in st.session_state:
    st.info("Choose settings and press Compute! to train a topic model.")
    st.stop()

analysis_key, model, dictionary, corpus, metadata, docs, original_docs = (
    st.session_state["lda_result"]
)
metadata_df = metadata.to_pandas()
labels = st.session_state["lda_labels"]
topic_names = [
    f"{topic}: {labels.get(topic) or ', '.join(word for word, _ in model.show_topic(topic, topn=3))}"
    for topic in range(model.num_topics)
]
metrics = st.columns(5)
for column, name, value in zip(
    metrics,
    ["Works", "Docs", "Authors", "Genres", "Topics"],
    [
        metadata_df[["author", "title"]].drop_duplicates().shape[0],
        len(docs),
        metadata_df.author.nunique(),
        metadata_df.genre.nunique(),
        model.num_topics,
    ],
):
    column.metric(name, value)

with st.expander("Document statistics"):
    st.plotly_chart(
        px.box(
            metadata_df,
            x="author",
            y="length",
            points="all",
            hover_data=["label", "docid"],
        )
    )
    st.plotly_chart(
        px.box(metadata_df, x="genre", y="length", hover_data=["label", "docid"])
    )

probabilities = create_dtm(
    analysis_key,
    model,
    model.num_topics,
    corpus,
    metadata_df.author.tolist(),
    collapsed=False,
).reset_index(drop=True)
probabilities.columns = topic_names
known = np.array([bool(bow) for bow in corpus])
author_topics = (
    probabilities.loc[known].groupby(metadata_df.loc[known, "author"]).mean()
)
genre_topics = probabilities.loc[known].groupby(metadata_df.loc[known, "genre"]).mean()
word_rows = [
    (topic, labels.get(topic, ""), word, probability)
    for topic in range(model.num_topics)
    for word, probability in model.show_topic(topic, topn=20)
]
word_table = pd.DataFrame(
    word_rows, columns=["topic_id", "label", "word", "probability"]
)

st.subheader("Topic words")
st.dataframe(word_table, hide_index=True)

st.subheader("Representative passages")
topic_id = st.selectbox(
    "Select topic",
    range(model.num_topics),
    format_func=lambda topic: topic_names[topic],
)
with st.form("topic_label"):
    label = st.text_input(
        "Topic label",
        value=labels.get(topic_id, ""),
        key=f"label_{analysis_key}_{topic_id}",
    )
    if st.form_submit_button("Save label"):
        labels[topic_id] = label.strip()
        st.rerun()
count = st.number_input(
    "Number of representative passages", min_value=1, max_value=20, value=5
)
ranking = probabilities.loc[known].iloc[:, topic_id].nlargest(int(count))
for row, probability in ranking.items():
    document = metadata_df.loc[row]
    st.markdown(
        f"**{document['author']} — {document['title']} — chunk {document['docid']} ({probability:.1%})**"
    )
    st.text("".join(original_docs[int(document["docid"])]))

st.subheader("Author and genre topic distributions")
author_tab, genre_tab = st.tabs(["Authors", "Genres"])
with author_tab:
    st.plotly_chart(create_dtm_heatmap(author_topics))
with genre_tab:
    st.caption(
        "Mean topic probabilities across chunks with vocabulary matches; each chunk has equal weight."
    )
    st.plotly_chart(create_dtm_heatmap(genre_topics))

st.subheader("Downloads")
document_table = pd.concat([metadata_df, probabilities], axis=1)
document_table["vocabulary_tokens"] = [sum(count for _, count in bow) for bow in corpus]
label_table = pd.DataFrame(
    {
        "topic_id": range(model.num_topics),
        "label": [labels.get(topic, "") for topic in range(model.num_topics)],
    }
)
for name, table, include_index in [
    ("Topic words", word_table, False),
    ("Document topics", document_table, False),
    ("Author topics", author_topics, True),
    ("Genre topics", genre_topics, True),
    ("Topic labels", label_table, False),
]:
    st.download_button(
        f"Download {name.lower()}",
        table.to_csv(index=include_index).encode("utf-8"),
        file_name=f"lda-{name.lower().replace(' ', '-')}.csv",
        mime="text/csv",
        on_click="ignore",
    )

with st.expander("PyLDAvis"):
    components.html(
        pyldavis_html(analysis_key, model, corpus, dictionary),
        width=1250,
        height=875,
        scrolling=True,
    )

st.subheader("文章の可視化")
author = st.selectbox("著者", sorted(metadata_df.author.unique()))
works = metadata_df.loc[metadata_df.author == author, "title"].unique()
work = st.selectbox("作品", sorted(works))
work_docids = metadata_df.loc[
    (metadata_df.author == author) & (metadata_df.title == work), "docid"
].tolist()
doc_id = st.selectbox("作品チャンク", work_docids)
components.html(
    colorize_topics(
        doc_id,
        dictionary,
        model,
        corpus,
        docs,
        original_docs,
        metadata_df.label.tolist(),
    ),
    height=500,
    scrolling=True,
)


def show_inference(text, name):
    processed, surfaces, bow = infer_text(text, get_tagger(), dictionary)
    if not bow:
        st.warning(
            "No vocabulary matches were found. Try a longer text related to this corpus."
        )
        return
    inferred = model.get_document_topics(bow, minimum_probability=0)
    inferred_table = pd.DataFrame(
        [(topic_names[topic], probability) for topic, probability in inferred],
        columns=["Topic", "Probability"],
    ).sort_values("Probability", ascending=False)
    st.dataframe(inferred_table, hide_index=True)
    components.html(
        colorize_topics(0, dictionary, model, [bow], [processed], [surfaces], [name]),
        height=500,
        scrolling=True,
    )


st.subheader("入力テキストのトピック推定")
with st.form("lda_inference"):
    query = st.text_area(
        "Text to infer topics",
        value="昔あるところに、美しい一人娘をお持ちの王さまとお妃さまがおりました。",
    )
    infer = st.form_submit_button("Infer topics")
if infer:
    show_inference(query, "Input text")

st.subheader("テキストファイルのトピック推定")
upload = st.file_uploader("UTF-8 text file", type=["txt"])
if upload is not None:
    try:
        show_inference(upload.getvalue().decode("utf-8"), upload.name)
    except UnicodeDecodeError:
        st.error("The file must be UTF-8 encoded text.")
