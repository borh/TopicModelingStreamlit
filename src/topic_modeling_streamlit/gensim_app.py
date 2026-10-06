import os
import socket
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import xxhash

from topic_modeling_streamlit.cache_locks import device_compute_lock
from topic_modeling_streamlit.data_lib import create_chunked_data, get_metadata
from topic_modeling_streamlit.gensim_lib import (
    colorize_topics,
    create_dtm,
    create_dtm_heatmap,
    get_tagger,
    infer_text,
    load_or_train_lda,
    parse_topic_labels,
    pyldavis_html,
)
from topic_modeling_streamlit.public_analyses import public_view, publish_button
from topic_modeling_streamlit.security import (
    MAX_LABEL_CHARS,
    MAX_TEXT_CHARS,
    csv_bytes,
    require_compute_access,
    validate_upload,
)

st.set_page_config(layout="wide")
public_view("gensim")
st.title("Gensim (LDA) を使用したトピックモデル")

DEFAULTS = {
    "Random state": 42,
    "Topics": 40,
    "Iterations": 2000,
    "Training batch size": 4000,
    "Passes": 15,
    "Chunk size (tokens)": 2000,
    "Min chunk size (tokens)": 20,
    "Chunks per work (0 = all)": 0,
    "Min document frequency": 5,
    "Max document frequency": 0.5,
    "Dictionary": "NOVEL",
    "Token form": "lemma",
    "Remove proper nouns": True,
    "Exclude parts of speech": [],
}
PRESETS = {
    "Full corpus": DEFAULTS,
    "Quick exploration": {
        **DEFAULTS,
        "Topics": 20,
        "Iterations": 100,
        "Passes": 2,
        "Chunk size (tokens)": 250,
        "Chunks per work (0 = all)": 5,
        "Min document frequency": 2,
        "Max document frequency": 0.9,
    },
}
for name, default in DEFAULTS.items():
    st.session_state.setdefault(f"lda_{name}", default)
preset = st.sidebar.selectbox("Settings preset", list(PRESETS))
if st.sidebar.button("Apply preset"):
    st.session_state.update(
        {f"lda_{name}": value for name, value in PRESETS[preset].items()}
    )


SETTING_LIMITS = {
    "Random state": 2**32 - 1,
    "Topics": 200,
    "Iterations": 2000,
    "Training batch size": 4000,
    "Passes": 20,
    "Chunk size (tokens)": 8000,
    "Min chunk size (tokens)": 8000,
    "Chunks per work (0 = all)": 1000,
    "Min document frequency": 50000,
}


def setting(name, minimum):
    return st.number_input(
        name, min_value=minimum, max_value=SETTING_LIMITS[name], key=f"lda_{name}"
    )


with st.sidebar.form("lda_settings"):
    st.subheader("LDA settings")
    random_state = setting("Random state", 0)
    num_topics = setting("Topics", 2)
    iterations = setting("Iterations", 1)
    chunksize = setting("Training batch size", 10)
    passes = setting("Passes", 1)
    st.subheader("Corpus settings")
    chunk_size = setting("Chunk size (tokens)", 10)
    min_chunk_size = setting("Min chunk size (tokens)", 1)
    chunks_per_work = setting("Chunks per work (0 = all)", 0)
    min_df = setting("Min document frequency", 1)
    max_df = st.number_input(
        "Max document frequency",
        min_value=0.01,
        max_value=1.0,
        key="lda_Max document frequency",
    )
    dictionary_name = st.selectbox(
        "Dictionary",
        ["NOVEL", "CWJ", "CSJ"],
        key="lda_Dictionary",
        format_func=lambda name: {
            "NOVEL": "UniDic-Novel",
            "CWJ": "UniDic-CWJ",
            "CSJ": "UniDic-CSJ",
        }[name],
    )
    token_form = st.selectbox("Token form", ["lemma", "surface"], key="lda_Token form")
    remove_proper_nouns = st.checkbox(
        "Remove proper nouns", key="lda_Remove proper nouns"
    )
    pos_filter = tuple(
        st.multiselect(
            "Exclude parts of speech",
            [
                "名詞",
                "動詞",
                "形容詞",
                "副詞",
                "助詞",
                "助動詞",
                "記号",
                "補助記号",
                "空白",
            ],
            key="lda_Exclude parts of speech",
        )
    )
    compute = st.form_submit_button("Compute!")
st.sidebar.caption(f"Running on {socket.gethostname()}")
with st.sidebar.expander("LDA model"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4d/Smoothed_LDA.png")
    st.markdown(
        "$\\theta$ is each document's topic distribution; $\\varphi$ is each topic's word distribution."
    )


def source_revision(dictionary_name) -> tuple[tuple[str, int, int], ...]:
    corpus_dir = Path("Aozora-Bunko-Fiction-Selection-2022-05-30")
    source_files = [
        corpus_dir / "groups.csv",
        *sorted((corpus_dir / "Plain").glob("*.txt")),
    ]
    dictionary_dir = os.environ.get(f"MECAB_DICDIR_{dictionary_name}")
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
    revision,
    tokenizer_version,
    chunk_size,
    min_chunk_size,
    chunks_per_work,
    preprocessing,
):
    with device_compute_lock("cpu"):
        return create_chunked_data(
            get_metadata(), chunk_size, min_chunk_size, chunks_per_work, **preprocessing
        )


@st.cache_data(max_entries=5, show_spinner="Loading or training LDA…")
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
    return load_or_train_lda(
        cache_key,
        _docs,
        min_df=min_df,
        max_df=max_df,
        random_state=random_state,
        num_topics=num_topics,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
    )


if compute:
    try:
        require_compute_access()
        if min_chunk_size > chunk_size:
            raise ValueError("Min chunk size must not exceed chunk size.")
        revision = source_revision(dictionary_name)
        preprocessing = {
            "dictionary": dictionary_name,
            "lemma": token_form == "lemma",
            "remove_proper_nouns": remove_proper_nouns,
            "pos_filter": pos_filter,
        }
        tokenizer_version = version("fugashi-plus")
        corpus_key = xxhash.xxh3_64_hexdigest(
            repr(
                (
                    revision,
                    tokenizer_version,
                    preprocessing,
                    chunk_size,
                    min_chunk_size,
                    chunks_per_work,
                )
            ).encode()
        )
        analysis_key = xxhash.xxh3_64_hexdigest(
            repr(
                (
                    "lda-cache-v1",
                    sys.version_info[:2],
                    version("gensim"),
                    version("numpy"),
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
            revision,
            tokenizer_version,
            chunk_size,
            min_chunk_size,
            chunks_per_work,
            preprocessing,
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
        previous_key = st.session_state.get("lda_result", (None,))[0]
        st.session_state["lda_result"] = (
            analysis_key,
            model,
            dictionary,
            corpus,
            metadata,
            docs,
            original_docs,
        )
        st.session_state["lda_preprocessing"] = preprocessing
        if previous_key != analysis_key:
            st.session_state["lda_labels"] = {}
            st.session_state["lda_label_revision"] = 0
    except (OSError, ValueError) as exc:
        st.error(str(exc))

if "lda_result" not in st.session_state:
    st.info(
        "Choose settings and press Compute! to train or reload a cached topic model."
    )
    st.stop()

analysis_key, model, dictionary, corpus, metadata, docs, original_docs = (
    st.session_state["lda_result"]
)
metadata_df = metadata.to_pandas()
labels = st.session_state["lda_labels"]
with st.expander("Restore or reset topic labels"):
    uploaded_labels = st.file_uploader(
        "Topic labels CSV", type=["csv"], key=f"labels_upload_{analysis_key}"
    )
    if (
        st.button("Import labels", disabled=uploaded_labels is None)
        and uploaded_labels is not None
    ):
        try:
            labels = parse_topic_labels(uploaded_labels.getvalue(), model.num_topics)
            st.session_state["lda_labels"] = labels
            st.session_state["lda_label_revision"] += 1
        except (ValueError, UnicodeDecodeError) as exc:
            st.error(str(exc))
    if st.button("Reset labels"):
        labels.clear()
        st.session_state["lda_label_revision"] += 1
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
    key=f"selected_topic_{analysis_key}",
    format_func=lambda topic: topic_names[topic],
)
with st.form("topic_label"):
    label = st.text_input(
        "Topic label",
        max_chars=MAX_LABEL_CHARS,
        value=labels.get(topic_id, ""),
        key=f"label_{analysis_key}_{st.session_state['lda_label_revision']}_{topic_id}",
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
exports = {}
for name, table, include_index in [
    ("Topic words", word_table, False),
    ("Document topics", document_table, False),
    ("Author topics", author_topics, True),
    ("Genre topics", genre_topics, True),
    ("Topic labels", label_table, False),
]:
    data = csv_bytes(table, index=include_index)
    exports[name.lower().replace(" ", "-")] = data
    st.download_button(
        f"Download {name.lower()}",
        data,
        file_name=f"lda-{name.lower().replace(' ', '-')}.csv",
        mime="text/csv",
        on_click="ignore",
    )

publish_button("gensim", analysis_key, exports["topic-words"], exports)

with st.expander("Topic similarity"):
    from sklearn.metrics.pairwise import cosine_similarity

    st.caption("Cosine similarity between topic-word distributions.")
    st.plotly_chart(
        px.imshow(
            cosine_similarity(model.get_topics()),
            x=topic_names,
            y=topic_names,
            zmin=0,
            zmax=1,
            aspect="auto",
        )
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
    preprocessing = st.session_state["lda_preprocessing"].copy()
    dictionary_name = preprocessing.pop("dictionary")
    tagger = get_tagger(dictionary_name)
    processed, surfaces, bow = infer_text(text, tagger, dictionary, **preprocessing)
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
        max_chars=MAX_TEXT_CHARS,
        value="昔あるところに、美しい一人娘をお持ちの王さまとお妃さまがおりました。",
    )
    infer = st.form_submit_button("Infer topics")
if infer:
    try:
        show_inference(query, "Input text")
    except ValueError as exc:
        st.error(str(exc))

st.subheader("テキストファイルのトピック推定")
upload = st.file_uploader("UTF-8 text file", type=["txt"])
if upload is not None:
    try:
        validate_upload(upload.getvalue())
        show_inference(upload.getvalue().decode("utf-8"), upload.name)
    except UnicodeDecodeError:
        st.error("The file must be UTF-8 encoded text.")
    except ValueError as exc:
        st.error(str(exc))
