import tqdm
import stqdm

tqdm.orignal_class = stqdm.stqdm

from bertopic import representation
import streamlit as st
from torch import embedding

st.set_page_config(layout="wide")

from data_lib import *
import pandas as pd
import re
from io import StringIO
from pathlib import Path


st.title("BERTopicを使用したトピックモデル")

st.sidebar.markdown("## 設定")

embedding_model = st.sidebar.selectbox(
    "モデル",
    [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "pkshatech/simcse-ja-bert-base-clcmlp",
        # "colorfulscoop/sbert-base-ja", # ValueError: zero-size array to reduction operation maximum which has no identity
    ],
)

representation_model = st.sidebar.selectbox(
    "Topic representation (ラベル解釈)",
    [
        None,
        "google/mt5-base",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "ChatGPT",
    ],
)

prompt = None
if representation_model and "/" in representation_model:
    prompt = st.sidebar.text_area(
        "Prompt",
        value="""トピックは次のキーワードで特徴付けられている: [KEYWORDS]. これらのキーワードを元に完結にトピックを次の通り要約する: """,
    )
    # [KEYWORDS]というキーワードを次の単語で表現する：

chunksize = st.sidebar.number_input("Maximum chunksize", min_value=10, value=100)
chunks = st.sidebar.number_input("Number of chunks per doc", min_value=10, value=50)

from bertopic import BERTopic
from bertopic.representation import OpenAI
from transformers import pipeline
from bertopic.representation import TextGeneration
from sklearn.feature_extraction.text import CountVectorizer

# As MeCab is not picklable, we do not cache any tokenization initialization
tagger = get_tagger()


def japanese_tokenizer(s):
    return tokenize(s, tagger)


def JapaneseCountVectorizer(ngram_range=(1, 1)):
    return CountVectorizer(
        ngram_range=ngram_range,
        tokenizer=japanese_tokenizer,
        lowercase=False,
        token_pattern=r".+",
    )


vectorizer = JapaneseCountVectorizer()

# import pysbd
#
# sentence_segmenter = pysbd.Segmenter(language="ja", clean=False)

all_metadata = get_metadata()


def split(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@st.cache_data
def create_corpus(chunksize: int, chunks: int):
    docs = []
    labels = []
    authors = []
    for file in Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob(
        "*.txt"
    ):
        title = file.stem
        with open(file, encoding="utf-8") as f:
            paragraphs = []
            for paragraph in f.read().splitlines():
                tokens = japanese_tokenizer(paragraph)
                if len(tokens) <= 8:
                    continue
                elif len(tokens) < chunksize:
                    paragraphs.append(paragraph)
                else:
                    xs = list(split(tokens, chunksize))
                    for i in range(len(xs)):
                        paragraphs.append("".join(xs[i]))
                    # sentences = sentence_segmenter.segment(paragraph)
                    # a, b = (
                    #     sentences[: len(sentences) // 2],
                    #     sentences[len(sentences) // 2 :],
                    # )
                    # paragraphs.append(japanese_tokenizer("".join(a)))
                    # paragraphs.append(japanese_tokenizer("".join(b)))

            paragraphs = paragraphs[:chunks]

            # if limit_paragraphs > 0:
            #     paragraphs = paragraphs[:limit_paragraphs]
            docs.extend(paragraphs)
            labels.extend([title for _ in range(len(paragraphs))])
            author = (
                all_metadata.query(f'filename == "{Path(file).name}"')
                .iloc[0, :]
                .author_ja
            )
            authors.extend([author for _ in range(len(paragraphs))])

    metadata = pd.DataFrame(
        {
            "authors": authors,
            "labels": labels,
            "lengths": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )
    return docs, labels, authors, metadata


docs, labels, authors, metadata = create_corpus(chunksize, chunks)

st.markdown(
    f"""
-   Authors: {", ".join(set(authors))}
-   Docs: {len(docs)}
-   Model: {embedding_model}
-   Representation: {representation_model}
"""
)

st.write(chunksizes_by_author_plot(metadata))

import openai


@st.cache_resource
def get_openai():
    with open("/run/agenix/openai-api") as f:
        openai.api_key = f.read().strip()
    representation_model = OpenAI(model="gpt-4", delay_in_seconds=10, chat=True)
    return representation_model


@st.cache_data
def create_bertopic_model(model, representation=False, prompt=prompt):
    if representation == "ChatGPT":
        representation_model = get_openai()
        topic_model = BERTopic(
            embedding_model=model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            verbose=True,
            calculate_probabilities=True,
        )
    elif representation and "/" in representation:
        if not prompt:
            prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
        generator = pipeline("text2text-generation", model=representation)
        representation_model = TextGeneration(generator, prompt=prompt)
        topic_model = BERTopic(
            embedding_model=model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            verbose=True,
            calculate_probabilities=True,
        )
    else:
        topic_model = BERTopic(
            embedding_model=model,
            vectorizer_model=vectorizer,
            verbose=True,
            calculate_probabilities=True,
        )
    return topic_model


@st.cache_data
def calculate_model(embedding_model, docs, representation=False, prompt=None):
    topic_model = create_bertopic_model(embedding_model, representation, prompt)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


topic_model, topics, probs = calculate_model(
    embedding_model, docs, representation_model, prompt
)

if "exported" not in st.session_state:
    st.session_state.exported = False


def toggle_export():
    st.session_state.exported = not st.session_state.exported


export = st.checkbox("モデルファイルのエキスポート", value=False, on_change=toggle_export)

from tempfile import NamedTemporaryFile

if st.session_state.exported:
    with NamedTemporaryFile(dir="temp") as tmp_file:
        topic_model.save(tmp_file.name)
        with open(tmp_file.name, "rb") as saved_model:
            st.download_button(
                "トピックモデルファイルのダウンロード", saved_model, on_click=toggle_export
            )

st.write(topic_model.visualize_topics())

st.write(topic_model.visualize_distribution(probs[0], min_probability=0.0))

if not representation_model:
    c01, c02 = st.columns(2)

    with c01:
        top_n_topics = st.number_input("Top n topics", min_value=1, value=10)
    with c02:
        n_words = st.number_input("n words", min_value=1, value=8)

    st.write(topic_model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words))


from sentence_transformers import SentenceTransformer
from umap import UMAP


@st.cache_data
def get_sentence_embeddings(docs, embedding_model):
    sentence_model = SentenceTransformer(embedding_model)
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    return embeddings


def visualize_docs(topic_model, docs, embeddings_model):
    embeddings = get_sentence_embeddings(docs, embedding_model)
    # Run the visualization with the original embeddings
    topic_model.visualize_documents(docs, embeddings=embeddings)

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(
        n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
    ).fit_transform(embeddings)
    return topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)


st.plotly_chart(visualize_docs(topic_model, docs, embedding_model))

st.plotly_chart(topic_model.visualize_hierarchy())

st.plotly_chart(topic_model.visualize_heatmap())


def visualize_text(topic_model, docs):
    # Calculate the topic distributions on a token-level
    topic_distr, topic_token_distr = topic_model.approximate_distribution(
        docs, calculate_tokens=True
    )

    # Visualize the token-level distributions
    df = topic_model.visualize_approximate_distribution(docs[0], topic_token_distr[0])
    return df


example_text = st.text_area("文章の可視化", value="どのようなトピックになるかなあ？")

st.write(visualize_text(topic_model, [example_text]))
