import streamlit as st
from stqdm import stqdm
import torch

import polars as pl
import plotly.express as px
from pathlib import Path
import socket
from typing import Iterator
from datetime import datetime, timedelta, timezone
import xxhash

import logging
from data_lib import is_katakana_sentence
import jaconv

from nlp_utils import LanguageProcessor, load_and_persist_model

st.set_page_config(layout="wide")

logging.getLogger("filelock").setLevel(logging.DEBUG)

st.title("BERTopic Playground")
st.markdown(
    """
-   Official documentation: <https://maartengr.github.io/BERTopic/index.html>
-   Github: <https://github.com/MaartenGr/BERTopic>
"""
)

st.sidebar.header("BERTopic Settings")


def get_embedding_model_options(language: str) -> list[str]:
    """Get embedding model options based on selected language."""
    if language == "Japanese":
        # Roughly ordered by JMTEB scores (calculated by me)
        return [
            "pkshatech/simcse-ja-bert-base-clcmlp",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-large",  # High JMTEB, but strange results (soft topic calculation not working)?
        ]
    # https://huggingface.co/spaces/mteb/leaderboard
    return [
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
        "thenlper/gte-base",
        "hkunlp/instructor-large",
    ]


def create_settings_form(language: str) -> st.form:
    """Create the settings form in the sidebar."""
    settings = st.sidebar.form("settings")

    tab1, tab2, tab3 = settings.tabs(["Model", "Tokenizer", "Corpus"])

    with tab1:
        st.selectbox(
            "Embedding model",
            get_embedding_model_options(language),
            key="embedding_model_option",
        )

        st.multiselect(
            "Topic representation",
            [
                "KeyBERTInspired",
                "MaximalMarginalRelevance",
                "PartOfSpeech",  # Needs spaCy
                "microsoft/Phi-3-small-8k-instruct",
                "GPT-4-0613",
            ],
            ["KeyBERTInspired"],
            # ["microsoft/Phi-3-small-8k-instruct"],
            key="representation_model_option",
        )
        with st.expander(
            "Explanation [(official documentation)](https://maartengr.github.io/BERTopic/getting_started/multiaspect/multiaspect.html)"
        ):
            st.markdown(
                """
            Representation models control how topics are labeled.
            In traditional LDA models, this might be a label such as `12_money_bank_business`, denoting the topic number and the three most frequent tokens in the topic.
            BERTopic offers multi-aspect representation where multiple representation can be computed at the same time.
            """
            )

    with tab2:
        tokenizer_dictionary_options = {
            "Japanese": {
                "MeCab": ["近現代口語小説UniDic", "UniDic-CWJ"],
                "Sudachi": [
                    "SudachiDict-full/A",
                    "SudachiDict-full/B",
                    "SudachiDict-full/C",
                ],
                "spaCy": ["ja_core_news_sm"],
                "Juman++": ["Jumandict"],
            },
            "English": {
                "spaCy": [
                    "en_core_web_sm",
                    "en_core_web_trf",
                ]
            },
        }

        st.selectbox(
            "Tokenizer and dictionary",
            [
                f"{tokenizer}/{dictionary}"
                for tokenizer, dictionaries in tokenizer_dictionary_options[
                    language
                ].items()
                for dictionary in dictionaries
            ],
            key="tokenizer_dictionary_option",
        )

        st.multiselect(
            "Default token representation (multiple selections will be formatted as surface/POS/... etc.)",
            ["orth", "lemma", "pos1"],
            ["orth"],
            key="tokenizer_features_option",
        )

        st.multiselect(
            "Remove POS",
            (
                [
                    "名詞",
                    "代名詞",
                    "形状詞",
                    "連体詞",
                    "副詞",
                    "接続詞",
                    "感動詞",
                    "動詞",
                    "形容詞",
                    "助動詞",
                    "助詞",
                    "接頭辞",
                    "接尾辞",
                    "記号",
                    "補助記号",
                    "空白",
                ]
                if language == "Japanese"
                else [
                    "ADJ",
                    "ADP",
                    "PUNCT",
                    "ADV",
                    "AUX",
                    "SYM",
                    "INTJ",
                    "CCONJ",
                    "X",
                    "NOUN",
                    "DET",
                    "PROPN",
                    "NUM",
                    "VERB",
                    "PART",
                    "PRON",
                    "SCONJ",
                ]
            ),
            key="tokenizer_pos_filter_option",
        )

        st.select_slider("N-gram range", range(1, 5), (1, 1), key="ngram_range_option")
        st.number_input(
            "Max document frequency", min_value=0.0, value=0.8, key="max_df_option"
        )

    with tab3:
        st.number_input(
            "Maximum chunksize", min_value=10, value=100, key="chunksize_option"
        )
        st.number_input(
            "Number of chunks per doc (0 for all)",
            min_value=0,
            value=50,
            key="chunks_option",
        )
        st.number_input(
            "Reduce to n topics (0 == auto (do not reduce))",
            min_value=0,
            value=0,
            key="nr_topics_option",
        )

    settings.radio(
        "Computation mode",
        [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"],
        key="device_option",
    )

    # FIXME migrate to phi
    if "representation_model_option" in st.session_state and any(
        "/" in opt for opt in st.session_state["representation_model_option"]
    ):
        settings.text_area(
            "Prompt",
            value="トピックは次のキーワードで特徴付けられている: [KEYWORDS]. これらのキーワードを元に完結にトピックを次の通り要約する: ",
            key="prompt_option",
            label_visibility="collapsed",
        )

    return settings


def set_options(reload: bool) -> None:
    """Set options in the session state."""
    for option in st.session_state.keys():
        if option.endswith("_option"):
            barename = option.replace("_option", "")
            st.session_state[barename] = st.session_state[option]
    st.session_state["tokenizer_type"] = (
        st.session_state.tokenizer_dictionary_option.split("/")[0]
    )
    st.session_state["dictionary_type"] = "/".join(
        st.session_state.tokenizer_dictionary_option.split("/")[1:]
    )
    st.session_state["prompt"] = st.session_state.get("prompt_option", "")
    if reload:
        st.session_state["reload"] = True

    unique_string = (
        "".join(
            f"{k}{v}" for k, v in sorted(st.session_state.items()) if k != "unique_id"
        )
        + language
    )
    st.session_state["unique_id"] = xxhash.xxh3_64_hexdigest(unique_string)
    st.toast("Compute!", icon="🎉")


language = st.sidebar.radio("Language", ("Japanese", "English"), 0, horizontal=True)
settings = create_settings_form(language)
submit_all = settings.form_submit_button("Compute!", on_click=set_options, args=(True,))

if "unique_id" not in st.session_state:
    set_options(False)

st.sidebar.markdown(
    f"""
### 参考文献
-   <https://github.com/MaartenGr/BERTopic>
-   <https://arxiv.org/abs/2203.05794>

Running on {socket.gethostname()}
"""
)


@st.cache_data
def get_metadata(language: str = "Japanese") -> pl.DataFrame | None:
    """Load metadata based on the selected language."""
    if language != "Japanese":
        return None

    metadata_df = pl.read_csv(
        "Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv", separator="\t"
    )
    return metadata_df


def split(lst: list[str], n: int) -> Iterator[list[str]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


all_metadata = get_metadata(language)


@st.cache_data
def create_corpus(
    _all_metadata: pl.DataFrame | None,
    language: str,
    chunksize: int,
    chunks: int,
    tokenizer_type: str,
    dictionary_type: str,
) -> tuple[list[str], pl.DataFrame]:
    """Create the corpus by chunking the documents."""
    docs = []
    labels = []
    filenames = []
    authors = []

    if language == "Japanese":
        tokenizer = LanguageProcessor(
            tokenizer_type=tokenizer_type,
            dictionary_type=dictionary_type,
            language=language,
        ).tokenizer
    else:
        tokenizer = LanguageProcessor(
            tokenizer_type=tokenizer_type,
            dictionary_type=dictionary_type,
            language=language,
            features=["text_with_ws"],
        ).tokenizer

    sep = "" if language == "Japanese" else " "

    files = list(
        Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob("*.txt")
        if language == "Japanese"
        else Path("./standard-ebooks-selection/").glob("*.txt")
    )

    for file in stqdm(files, desc="Chunking corpus"):
        title = file.stem
        with open(file, encoding="utf-8") as f:
            doc, tokens_chunk, running_count = [], [], 0
            for paragraph in f.read().splitlines():
                if chunks > 0 and running_count >= chunks:
                    break
                if not paragraph:
                    continue
                if language == "Japanese" and is_katakana_sentence(paragraph):
                    logging.warning(
                        f"Converting katakana sentence to hiragana: {paragraph} -> {jaconv.kata2hira(paragraph)}"
                    )
                    paragraph = jaconv.kata2hira(paragraph)

                tokens = (
                    tokenizer.tokenize(paragraph)
                    if language == "Japanese"
                    else paragraph.split()
                )

                if len(tokens) >= chunksize:
                    doc.append(sep.join(tokens_chunk))
                    running_count += 1
                    xs = list(split(tokens, chunksize))
                    last_chunk = xs.pop()
                    doc.extend(sep.join(x) for x in xs)
                    running_count += len(xs)
                    tokens_chunk = last_chunk if len(last_chunk) < chunksize else []
                elif len(tokens) + len(tokens_chunk) > chunksize:
                    doc.append(sep.join(tokens_chunk))
                    running_count += 1
                    tokens_chunk = tokens
                else:
                    tokens_chunk.extend(tokens)

            if tokens_chunk:
                doc.append(sep.join(tokens_chunk))

            if chunks > 0:
                doc = doc[:chunks]

            docs.extend(doc)
            labels.extend([title for _ in range(len(doc))])
            filenames.extend([file.name for _ in range(len(doc))])
            author = (
                title.split("_")[0]
                if _all_metadata is None
                else _all_metadata.filter(pl.col("filename") == Path(file).name)
                .select(pl.col("author_ja"))
                .head(1)
                .to_series()
                .to_list()[0]
            )
            authors.extend([author for _ in range(len(doc))])

    metadata = pl.DataFrame(
        {
            "author": authors,
            "label": labels,
            "filename": filenames,
            "length": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )

    if all_metadata is None:
        metadata = metadata.with_columns(title=pl.col("label"), genre=pl.lit("all"))
    elif not all_metadata.is_empty():
        metadata = all_metadata.select(
            pl.col("filename", "genre", "year"),
            pl.col("author_ja").alias("author"),
            pl.col("title_ja").alias("title"),
        ).join(
            metadata.select(pl.col("filename", "label", "docid", "length")),
            on="filename",
        )
    return docs, metadata


docs, metadata = create_corpus(
    all_metadata,
    language,
    st.session_state.chunksize,
    st.session_state.chunks,
    st.session_state.tokenizer_type,
    st.session_state.dictionary_type,
)

authors = set(
    metadata.select(pl.col("author"))
    .unique()
    .sort(pl.col("author"))
    .get_column("author")
)

with st.expander("Debug information"):
    st.write(st.session_state)
    st.markdown(
        f"""
    -   Authors: {", ".join(authors)}
    -   Works: {metadata.select(pl.col('title')).n_unique()}
    -   Docs: {len(docs)}
        - Sample: {docs[0][:100]}
    -   Chunksize: {st.session_state.chunksize}
    -   Chunks/doc: {st.session_state.chunks}
    -   Topic reduction (limit): {st.session_state.nr_topics}
    -   Model: {st.session_state.embedding_model}
    -   Representation: {st.session_state.representation_model}
    """
    )


def chunksizes_by_author_plot(metadata):
    return px.box(
        metadata.to_pandas(),
        x="author",
        y="length",
        hover_data=["label", "docid"],
        title="Author document length distribution",
    )


if not metadata.is_empty():
    with st.expander("Open to see basic document stats"):
        st.write(chunksizes_by_author_plot(metadata))
        st.write(
            px.histogram(
                metadata.to_pandas(),
                x="author",
                title="Author document distribution",
            )
        )
        st.write(
            px.box(
                metadata.to_pandas(),
                x="genre",
                y="length",
                hover_data=["label", "docid"],
                title="Genre document length distribution",
            )
        )
        st.write(
            px.histogram(
                metadata.to_pandas(),
                x="genre",
                title="Genre document distribution",
            )
        )

(topic_model_path, topic_model, embeddings, reduced_embeddings, topics, probs) = (
    load_and_persist_model(
        docs,
        language,
        st.session_state.embedding_model,
        st.session_state.representation_model,
        st.session_state.prompt,
        st.session_state.nr_topics,
        st.session_state.tokenizer_type,
        st.session_state.dictionary_type,
        st.session_state.ngram_range,
        st.session_state.max_df,
        st.session_state.tokenizer_features,
        st.session_state.tokenizer_pos_filter,
        device=st.session_state.device,
    )
)

if topic_model.nr_topics != st.session_state.nr_topics:
    logging.warning(f"Reducing topics to {st.session_state.nr_topics}...")
    topic_model.reduce_topics(docs, st.session_state.nr_topics)
    (topic_model_path, topic_model, embeddings, reduced_embeddings, topics, probs) = (
        load_and_persist_model(
            docs,
            language,
            st.session_state.embedding_model,
            st.session_state.representation_model,
            st.session_state.prompt,
            st.session_state.nr_topics,
            st.session_state.tokenizer_type,
            st.session_state.dictionary_type,
            st.session_state.ngram_range,
            st.session_state.max_df,
            st.session_state.tokenizer_features,
            st.session_state.tokenizer_pos_filter,
            topic_model=topic_model,
            embeddings=embeddings,
            reduced_embeddings=reduced_embeddings,
            topics=topic_model.topics_,
            probs=topic_model.probs_,
            device=st.session_state.device,
        )
    )
elif (
    topic_model.nr_topics != st.session_state.nr_topics
    and topic_model.nr_topics is not None
):
    logging.warning(f"Loading model on nr_topics change: {st.session_state.nr_topics}")
    if st.session_state.nr_topics is None:
        st.session_state.nr_topics = "auto"
    (topic_model_path, topic_model, embeddings, reduced_embeddings, topics, probs) = (
        load_and_persist_model(
            docs,
            language,
            st.session_state.embedding_model,
            st.session_state.representation_model,
            st.session_state.prompt,
            st.session_state.nr_topics,
            st.session_state.tokenizer_type,
            st.session_state.dictionary_type,
            st.session_state.ngram_range,
            st.session_state.max_df,
            st.session_state.tokenizer_features,
            st.session_state.tokenizer_pos_filter,
            device=st.session_state.device,
        )
    )

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Works", value=metadata.select(pl.col("title")).n_unique())
mc2.metric("Docs", value=len(docs))
mc3.metric("Authors", value=len(authors))
mc4.metric("Genres", value=metadata.select(pl.col("genre")).n_unique())
mc5.metric("Topics", value=len(topic_model.topic_sizes_))

model_creation_time = datetime.fromtimestamp(
    topic_model_path.stat().st_mtime, tz=timezone.utc
).astimezone(timezone(timedelta(hours=9)))

st.sidebar.write(
    f"Model {topic_model_path} created on {model_creation_time:%Y-%m-%d %H:%M}"
)


@st.cache_data
def cached_visualize_topics(unique_id):
    return topic_model.visualize_topics()


st.write(cached_visualize_topics(st.session_state.unique_id))

"# Corpus topic browser"

selection_col1, selection_col2 = st.columns(2)

with selection_col1:
    selected_author = st.selectbox("著者", options=authors)

author_works = set(
    metadata.filter(pl.col("author") == selected_author)
    .select(pl.col("title"))
    .unique()
    .sort(pl.col("title"))
    .to_series()
    .to_list()
)

with selection_col2:
    selected_work = st.selectbox("作品", options=author_works)

work_docids = (
    metadata.filter(pl.col("title") == selected_work)
    .select(pl.col("docid"))
    .to_series()
    .to_list()
)

# Avoid serializing and sending all docids to client; instead use slide min and max values.
# This works because docids are sequential per author per work.
if work_docids:
    doc_id = st.slider("作品チャンク", min(work_docids), max(work_docids))
    st.markdown(f"> {docs[doc_id]}")
    st.write(topic_model.visualize_distribution(probs[doc_id], min_probability=0.0))
    with st.expander(
        "Explanation (Official documentation)[https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html]"
    ):
        st.markdown(
            """The topic distributions presented here are approximated using a sliding window over the document. For each window, its c-TF-IDF representation is used to find out how similar it is to all topics, and the final distribution is a sum over all windows."""
        )

if not st.session_state.representation_model:
    st.markdown("# Topic word browser")

    c01, c02 = st.columns(2)

    with c01:
        top_n_topics = st.number_input("Top n topics", min_value=1, value=10)
    with c02:
        n_words = st.number_input("n words", min_value=1, value=8)

    st.write(topic_model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words))


"# Document and topic 2D plot (UMAP)"


@st.cache_data
def visualize_docs(unique_id, docs, reduced_embeddings):
    return topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)


st.plotly_chart(visualize_docs(st.session_state.unique_id, docs, reduced_embeddings))


@st.cache_data
def get_topic_info(unique_id):
    return topic_model.get_topic_info()[
        [
            "Topic",
            "Count",
            "Name",
            "Representation",
        ]
    ]


with st.expander("Topic statistics"):
    st.write(get_topic_info(st.session_state.unique_id))


"# Topic information"

ttab1, ttab2, ttab3, ttab4 = st.tabs(
    [
        "Hierarchical cluster analysis",
        "Similarity matrix heatmap",
        "Per-genre topic distribution",
        "DataMapPlot",
    ]
)


@st.cache_data
def visualize_datamap(unique_id):
    return topic_model.visualize_document_datamap(
        docs,
        darkmode=True,
    )


ttab1.plotly_chart(topic_model.visualize_hierarchy())
ttab2.plotly_chart(topic_model.visualize_heatmap())

if not metadata.is_empty():
    topics_per_class = topic_model.topics_per_class(
        docs, classes=metadata.get_column("genre").to_list()
    )
    ttab3.plotly_chart(
        topic_model.visualize_topics_per_class(
            topics_per_class, top_n_topics=len(topic_model.topic_sizes_)
        )
    )

ttab4.plotly_chart(visualize_datamap(st.session_state.unique_id))

st.markdown("# Topic query")

topic_query = st.text_input("Find topics")

if topic_query != "":
    qtopics, qsim = topic_model.find_topics(topic_query)
    st.write(
        pl.DataFrame(
            {
                "Topic": qtopics,
                "Representation": [
                    str(topic_model.get_topic_info(topic_id)["Name"])
                    for topic_id in qtopics
                ],
                "Similarity": qsim,
            }
        ).to_pandas()
    )

"# Topic inference"


def visualize_text(unique_id, doc, use_embedding_model):
    """Wrapper for token-level topic visualization. Currently, using the cf-tfifdf model is broken, so we use the embedding model."""
    _topic_distr, topic_token_distr = topic_model.approximate_distribution(
        doc,
        use_embedding_model=use_embedding_model,
        calculate_tokens=True,
        min_similarity=0.001,
        separator="",  # No whitespace for Japanese
    )

    df = topic_model.visualize_approximate_distribution(doc, topic_token_distr[0])
    return df


use_embedding_model_option = st.checkbox("Use embedding model", False)

# https://www.aozora.gr.jp/cards/002231/files/62105_76819.html
# https://mysterytribune.com/suspense-novel-excerpt-the-echo-killing-by-christi-daugherty/
example_text = st.text_area(
    f"Token topic approximation using {'embedding' if use_embedding_model_option else 'c-tf-idf'} model",
    value=(
        """金の羽根
昔あるところに、月にもお日さまにも増して美しい一人娘をお持ちの王さまとお妃さまがおりました。娘はたいそうおてんばで、宮殿中の物をひっくり返しては大騒ぎをしていました。"""
        if language == "Japanese"
        else """It was one of those nights.
Early on there was a flicker of hope— a couple of stabbings, a car wreck with potential. But the wounds weren’t serious and the accident was routine. After that it fell quiet."""
    ),
)


st.write(
    visualize_text(st.session_state.unique_id, example_text, use_embedding_model_option)
)
