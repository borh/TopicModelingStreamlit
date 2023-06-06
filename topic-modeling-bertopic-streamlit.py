import streamlit as st
from stqdm import stqdm
from sentence_transformers import SentenceTransformer
from umap import UMAP
import pysbd
from bertopic import BERTopic
import torch
from bertopic.representation import (
    # TextGeneration,
    # BaseRepresentation,
    KeyBERTInspired,
    MaximalMarginalRelevance,
)
from sklearn.feature_extraction.text import CountVectorizer
import rhoknp
from fugashi import Tagger, GenericTagger, create_feature_wrapper
import sudachipy
import spacy
import pandas as pd
import polars as pl
import plotly.express as px
from pathlib import Path
import re
import os
import socket
from typing import Iterator
from datetime import datetime, timedelta, timezone
import pickle
import cloudpickle
import xxhash
from filelock import FileLock
from time import sleep
import logging
from data_lib import is_katakana_sentence
import jaconv

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


language = st.sidebar.radio(
    "Language",
    ("Japanese", "English"),
    0,
    horizontal=True,
)

settings = st.sidebar.form("settings")

tab1, tab2, tab3 = settings.tabs(["Model", "Tokenizer", "Corpus"])

with tab1:
    embedding_model_option = st.selectbox(
        "Embedding model",
        [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "pkshatech/simcse-ja-bert-base-clcmlp",
        ]
        if language == "Japanese"
        else [
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
        ],
    )

    representation_model_option = st.multiselect(  # Aspects -> multiselectbox
        "Topic representation",
        [
            "KeyBERTInspired",
            "MaximalMarginalRelevance",
            "PartOfSpeech",  # Needs spaCy
            # None,  # Default: most frequent tokens
            # "google/mt5-base",
            # "rinna/japanese-gpt-neox-3.6b-instruction-sft",
            # "ChatGPT",
        ],
        ["KeyBERTInspired", "MaximalMarginalRelevance"],
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
    # prompt_option = None
    # if representation_model_option and "/" in representation_model_option:
    #     prompt_option = settings.text_area(
    #         "Prompt",
    #         value="""トピックは次のキーワードで特徴付けられている: [KEYWORDS]. これらのキーワードを元に完結にトピックを次の通り要約する: """,
    #         label_visibility="collapsed",
    #     )
    #     # [KEYWORDS]というキーワードを次の単語で表現する：

    tdu = {
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

    tokenizer_dictionary_option = st.selectbox(
        "Tokenizer and dictionary",
        [
            f"{tokenizer}/{dictionary}"
            for tokenizer, dictionaries in tdu[language].items()
            for dictionary in dictionaries
        ],
    )

    tokenizer_features_option = st.multiselect(
        "Tokenizer: default features to use as token (multiple features will be transformed to surface/POS/... format)",
        [
            "orth",
            "lemma",
            "pos1",
        ],
        ["orth"],
    )

    tokenizer_pos_filter_option = set(
        st.multiselect(
            "Tokenizer: remove selected POS from c-tf-idf",
            ["symbols/punctuation", "adjectives", "adverbs", "numerals"],
        )
    )

    ngram_range_option = st.select_slider("N-gram range", range(1, 5), (1, 1))


with tab3:
    chunksize_option = st.number_input(
        "Maximum chunksize",
        min_value=10,
        value=100,
    )
    chunks_option = st.number_input(
        "Number of chunks per doc (0 for all)",
        min_value=0,
        value=50,
    )

    nr_topics_option = st.number_input(
        "Reduce to n topics (0 == auto (do not reduce))",
        min_value=0,
        value=0,
    )
    if nr_topics_option == 0:
        nr_topics_option = "auto"


device_option = settings.radio(
    "Computation mode",
    [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"],
)


def set_options(reload):
    for option in [
        "embedding_model_option",
        "representation_model_option",
        # "prompt_option",
        "tokenizer_features_option",
        "tokenizer_pos_filter_option",
        "ngram_range_option",
        "chunksize_option",
        "chunks_option",
        "nr_topics_option",
        "device_option",
    ]:
        barename = option.replace("_option", "")
        st.session_state[barename] = globals()[option]
    st.session_state["tokenizer_type"] = tokenizer_dictionary_option.split("/")[0]
    st.session_state["dictionary_type"] = "/".join(
        tokenizer_dictionary_option.split("/")[1:]
    )
    if reload:
        st.session_state["reload"] = True

    unique_string = (
        "".join(
            f"{k}{v}" for k, v in sorted(st.session_state.items()) if k != "unique_id"
        )
        + language
    )
    st.session_state["unique_id"] = xxhash.xxh3_64_hexdigest(unique_string)


submit_all = settings.form_submit_button("Compute!", on_click=set_options, args=(True,))

# # We need to set options once, to initialize
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
def get_metadata() -> pl.DataFrame:
    metadata_df = pl.read_csv(
        "Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv", separator="\t"
    )
    return metadata_df


def split(lst: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


CustomFeatures = create_feature_wrapper(
    "CustomFeatures",
    (
        "pos1 pos2 pos3 pos4 cType cForm lForm lemma orth pron "
        "pronBase goshu orthBase iType iForm fType fForm "
        "kana kanaBase form formBase iConType fConType aType "
        "aConType lid lemma_id"
    ).split(" "),
)


class JapaneseTagger:
    def __init__(
        self,
        tokenizer_type: str,
        dictionary_type: str,
        ngram_range: tuple[int, int] = (1, 1),
        features: list[str] | None = None,
        pos_filter: set[str] | None = None,
        language: str = "Japanese",
    ):
        self.tokenizer_type = tokenizer_type
        self.dictionary_type = dictionary_type
        self.ngram_range = ngram_range
        self.features = features
        self.pos_filter = pos_filter
        self.sentence_segmenter = pysbd.Segmenter(
            language="ja" if language == "Japanese" else "en", clean=False
        )
        self.whitespace_rx = re.compile(r"^\s*$")
        if tokenizer_type == "MeCab":
            self.pos_filter_fn = (
                lambda t: t.pos in self.pos_filter if self.pos_filter else True
            )
            if dictionary_type == "近現代口語小説UniDic":
                self.tagger = GenericTagger(
                    "-r ./65_novel/dicrc -d ./65_novel/ -Owakati",
                    wrapper=CustomFeatures,
                )
            else:
                # TODO other dictionaries
                dic_path = os.environ["UNIDIC_CWJ_PATH"]
                self.tagger = Tagger(f"-d {dic_path}")
            self.tokenizer_fn = self._mecab_tokenizer_fn
        elif tokenizer_type == "Sudachi":
            sudachi = sudachipy.Dictionary(dict="full").create()

            if dictionary_type.endswith("A"):
                self.mode = sudachipy.SplitMode.A
            elif dictionary_type.endswith("B"):
                self.mode = sudachipy.SplitMode.B
            elif dictionary_type.endswith("C"):
                self.mode = sudachipy.SplitMode.C
            else:
                raise Exception(f"Unknown dictionary: {dictionary_type}")
            self.tagger = sudachi.tokenize
            self.tokenizer_fn = self._sudachi_tokenizer_fn
        elif tokenizer_type == "spaCy":
            nlp = spacy.load(dictionary_type)
            self.tagger = nlp
            self.tokenizer_fn = self._spacy_tokenizer_fn
        elif tokenizer_type == "Juman++":
            jumanpp = rhoknp.Jumanpp()
            self.tagger = jumanpp.apply_to_sentence
            self.tokenizer_fn = self._jumanpp_tokenizer_fn

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            tokenizer=self.tokenizer_fn,
            lowercase=False if language == "Japanese" else True,
            token_pattern=r".+",
        )

    def _mecab_features_fn(self, t) -> str:
        mapping = {
            "orth": lambda t: t.surface,
            "pos1": lambda t: t.feature.pos,
            # Unknown/OOV words will get surface form returned
            "lemma": lambda t: t.feature.lemma or t.surface,
        }
        if not self.features:
            return t.surface
        else:
            return "/".join(mapping[feature](t) for feature in self.features)

    def _mecab_tokenizer_fn(self, s) -> list[str]:
        return [
            self._mecab_features_fn(t)
            for sentence in self.sentence_segmenter.segment(s)
            for t in self.tagger(sentence)
            if not self.whitespace_rx.match(t.surface) or not self.pos_filter_fn(t)
        ]

    def _sudachi_features_fn(self, t) -> str:
        mapping = {
            "orth": lambda t: t.surface(),
            "pos1": lambda t: t.part_of_speech()[0],
            "lemma": lambda t: t.normalized_form(),
        }
        if not self.features:
            return t.surface()
        else:
            return "/".join(mapping[feature](t) for feature in self.features)

    def _sudachi_tokenizer_fn(self, s) -> list[str]:
        sentences = self.sentence_segmenter.segment(s)
        tokens: list[str] = []
        for sentence in sentences:
            try:
                tokens.extend(
                    self._sudachi_features_fn(t)
                    for t in self.tagger(sentence, self.mode)
                    if not self.whitespace_rx.match(t.surface())
                    or (
                        self.pos_filter and not t.part_of_speech()[0] in self.pos_filter
                    )
                )
            except Exception:
                logging.warning(f"Sentence too long for analysis: {sentence}")
                # Limit maximum sentence length to a reasonable amount (around 2000)
                token_splits = split(sentence, 2000)
                tokens.extend(
                    self._sudachi_features_fn(t)
                    for ts in token_splits
                    for t in self.tagger(ts, self.mode)
                    if not self.whitespace_rx.match(t.surface())
                    or (
                        self.pos_filter and not t.part_of_speech()[0] in self.pos_filter
                    )
                )
        return tokens

    def _spacy_tokenizer_fn(self, s) -> list[str]:
        mapping = {
            "orth": lambda t: t.orth_,
            "pos1": lambda t: t.pos_,
            "lemma": lambda t: t.lemma_,
            "text_with_ws": lambda t: t.text_with_ws,
        }
        sentences = self.sentence_segmenter.segment(s)
        return [
            "/".join(mapping[feature](t) for feature in self.features)
            for doc in self.tagger.pipe(sentences, disable=["parser", "ner"])
            for t in doc
            if not self.whitespace_rx.match(t.orth_)
            and not (self.pos_filter and t.pos_ in self.pos_filter)
        ]

    def _jumanpp_features_fn(self, t) -> str:
        mapping = {
            "orth": lambda t: t.text,
            "pos1": lambda t: t.pos,
            "lemma": lambda t: t.lemma,
        }
        if not self.features:
            return t.text
        else:
            return "/".join(mapping[feature](t) for feature in self.features)

    def _jumanpp_tokenizer_fn(self, s) -> list[str]:
        tokens = [
            self._jumanpp_features_fn(t)
            for sentence in self.sentence_segmenter.segment(s)
            for t in self.tagger(sentence).morphemes
            if not self.whitespace_rx.match(t.text)
            and not (self.pos_filter and t.pos in self.pos_filter)
        ]
        return tokens

    def __getstate__(self):
        logging.error("getstate")
        return (
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.features,
            self.pos_filter,
            self._mecab_tokenizer_fn,
            self._sudachi_tokenizer_fn,
            self._jumanpp_tokenizer_fn,
        )

    def __setstate__(self, state):
        logging.error("setstate")
        (
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.features,
            self.pos_filter,
            self._mecab_tokenizer_fn,
            self._sudachi_tokenizer_fn,
            self._jumanpp_tokenizer_fn,
        ) = state
        # Restore from initial params
        self.__init__(
            self.tokenizer_type,
            self.dictionary_type,
            ngram_range=self.ngram_range,
            features=self.features,
            pos_filter=self.pos_filter,
        )

    # def __reduce__(self):
    #    logging.error("reduce")
    #    return (
    #        type(self),
    #        (
    #            self.tokenizer_type,
    #            self.dictionary_type,
    #            self.ngram_range,
    #            self._mecab_tokenizer_fn,
    #            self._sudachi_tokenizer_fn,
    #            self._jumanpp_tokenizer_fn,
    #        ),
    #    )

    # def __reduce_ex__(self, protocol):
    #     logging.error("reduce_ex")
    #     return (
    #         type(self),
    #         (
    #             self.tokenizer_type,
    #             self.dictionary_type,
    #             self.ngram_range,
    #             self._mecab_tokenizer_fn,
    #             self._sudachi_tokenizer_fn,
    #             self._jumanpp_tokenizer_fn,
    #         ),
    #     )

    # def tokenize(self, s):
    #     sentences = self.sentence_segmenter.segment(s)
    #     return [t for sentence in sentences for t in self.tokenizer_fn(sentence)]


all_metadata = get_metadata()


# Note that we do not take token filtering or features into account in corpus creation
# as this is also used for embedding model.
@st.cache_data
def create_corpus(
    _all_metadata,
    language: str,
    chunksize: int,
    chunks: int,
    tokenizer_type: str,
    dictionary_type: str,
) -> tuple[list[str], pl.DataFrame]:
    docs = []
    labels = []
    filenames = []
    authors = []
    if language == "Japanese":
        tokenizer = JapaneseTagger(tokenizer_type, dictionary_type, language=language)
        sep = ""
    else:
        # We use spaCy's Token.text_with_ws to be able to use same logic as with
        # Japanese version when joining back tokens.
        tokenizer = JapaneseTagger(
            tokenizer_type,
            dictionary_type,
            language=language,
            features=["text_with_ws"],
        )
        sep = " "

    files = list(
        Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob("*.txt")
        if language == "Japanese"
        else Path("./standard-ebooks-selection/").glob("*.txt")
    )
    for file in stqdm(files, desc="Chunking corpus"):
        logging.error(file)
        title = file.stem
        with open(file, encoding="utf-8") as f:
            # "doc": list of units of analysis (~paragraph-size text) to pass to embedding model
            doc: list[str] = []
            # Temporary container for tokens to put in chunk
            tokens_chunk: list[str] = []
            running_count = 0
            for paragraph in f.read().splitlines():
                if chunks > 0 and running_count >= chunks:
                    logging.warning("Ending early.")
                    break
                if paragraph == "":
                    continue
                if language == "Japanese" and is_katakana_sentence(paragraph):
                    paragraph = jaconv.kata2hira(paragraph)
                if language == "Japanese":
                    tokens = tokenizer.tokenizer_fn(paragraph)
                else:
                    tokens = paragraph.split()
                logging.error(f"tokens: {len(tokens)}")
                if len(tokens) >= chunksize:
                    # Add current tokens_chunk
                    doc.append(sep.join(tokens_chunk))
                    running_count += 1
                    # Split tokens into chunksize-size chunks
                    xs = list(split(tokens, chunksize))
                    last_chunk = xs.pop()
                    doc.extend(sep.join(x) for x in xs)
                    running_count += len(xs)
                    if len(last_chunk) < chunksize:
                        tokens_chunk = last_chunk
                    else:
                        tokens_chunk = []
                # If adding paragraph to chunk goes over chunksize, commit current chunk to paragraphs and init new chunk with paragraph
                elif len(tokens) + len(tokens_chunk) > chunksize:
                    doc.append(sep.join(tokens_chunk))
                    running_count += 1
                    tokens_chunk = tokens
                # Otherwise, add to chunk
                else:
                    tokens_chunk.extend(tokens)
            # Add leftover (partial) chunk to paragraphs
            if tokens_chunk:
                doc.append(sep.join(tokens_chunk))

            # A chunks value of 0 returns all data chunks
            if chunks > 0:
                doc = doc[:chunks]

            docs.extend(doc)
            labels.extend([title for _ in range(len(doc))])
            filenames.extend([file.name for _ in range(len(doc))])
            author = (
                _all_metadata.filter(pl.col("filename") == Path(file).name)
                .select(pl.col("author_ja"))
                .head(1)[0]
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

    if not all_metadata.is_empty():
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
    fig = px.box(
        metadata.to_pandas(),
        x="author",
        y="length",
        # points="all", # Too heavy for large amounts
        hover_data=["label", "docid"],  # color="author",
        title="Author document distribution",
    )
    return fig


if not metadata.is_empty():
    with st.expander("Open to see basic document stats"):
        st.write(chunksizes_by_author_plot(metadata))
        st.write(
            px.box(
                metadata.to_pandas(),
                x="genre",
                y="length",
                hover_data=["label", "docid"],
                title="Genre document distribution",
            )
        )

# import openai
#
#
# @st.cache_resource
# def get_openai():
#     with open("/run/agenix/openai-api") as f:
#         openai.api_key = f.read().strip()
#     representation_model = OpenAI(model="gpt-4", delay_in_seconds=10, chat=True)
#     return representation_model


# FIXME Caching:
# https://github.com/streamlit/streamlit/issues/6295
# So our strategy must be:
# -   wherever possible, use st.cache_data
# -   for models and unhashable inputs, use surrogate indicators of change to trigger reloads (?!)
#     for this, we would need a custom model load/check function that intelligently caches, loads and saves on changes.


# @st.cache_resource
def calculate_model(
    docs: list[str],
    vectorizer: CountVectorizer,
    embedding_model: str,
    representation_model: list[str],
    prompt: str | None,
    nr_topics: str | int,
):
    if representation_model:
        representation_models = {}

        # if representation_model == "ChatGPT":
        #     representation_model = get_openai()
        representation_models["Main"] = KeyBERTInspired()
        if "KeyBERTInspired" in representation_model:
            representation_models["KeyBERTInspired"] = KeyBERTInspired()
        if "MaximalMarginalRelevance" in representation_model:
            representation_models[
                "MaximalMarginalRelevance"
            ] = MaximalMarginalRelevance(diversity=0.5)
        # Only with spaCy:
        # elif representation_model == "PartOfSpeech":
        #     representation_model = representation.PartOfSpeech()
        # elif representation_model and "/" in representation_model:
        #     if not prompt:
        #         prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
        #     generator = pipeline("text2text-generation", model=representation_model)
        #     representation_model = TextGeneration(generator, prompt=prompt)
        # else:
        #     raise Exception(f"Unsupported representation model {representation_model}")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            representation_model=representation_models,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )
    else:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )

    sentence_model = SentenceTransformer(
        embedding_model, device=st.session_state.device
    )
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    topic_model = topic_model.fit(docs, embeddings)
    topics, probs = topic_model.transform(docs, embeddings)
    reduced_embeddings = UMAP(
        n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
    ).fit_transform(embeddings)

    return topic_model, embeddings, reduced_embeddings, topics, probs


def save_computed(
    path, vectorizer_model, topics, probs, embeddings, reduced_embeddings
):
    with open(f"{path}-vectorizer_model.pickle", "wb") as f:
        cloudpickle.dump(vectorizer_model, f)
    with open(f"{path}-topics.pickle", "wb") as f:
        pickle.dump(topics, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-probs.pickle", "wb") as f:
        pickle.dump(probs, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-reduced_embeddings.pickle", "wb") as f:
        pickle.dump(reduced_embeddings, f, pickle.HIGHEST_PROTOCOL)
    Path(f"{path}.finished").touch()


# As long as this function does not do any unneccessary work, it should be the same as @st.cache or better
def load_and_persist_model(
    docs: list[str],
    language: str,
    embedding_model: str,
    representation_model: list[str],
    prompt: str | None,
    nr_topics: str | int,
    tokenizer_type: str,
    dictionary_type: str,
    ngram_range: tuple[int, int],
    tokenizer_features: list[str],
    tokenizer_pos_filter: set[str] | None,
    topic_model=None,
    embeddings=None,
    reduced_embeddings=None,
    topics=None,
    probs=None,
):
    docs_hash = xxhash.xxh3_64_hexdigest(pickle.dumps(docs, pickle.HIGHEST_PROTOCOL))
    settings_hash = xxhash.xxh3_64_hexdigest(
        pickle.dumps(
            (
                representation_model,
                prompt,
                nr_topics,
                tokenizer_type,
                dictionary_type,
                ngram_range,
                tokenizer_features,
                tokenizer_pos_filter,
            ),
            pickle.HIGHEST_PROTOCOL,
        )
    )
    # representation_id = ""
    # if representation_model:
    #     representation_id = f"{'_'.join(k.replace('/', '_') for k in representation_model)}-{xxhash.xxh3_64_hexdigest(prompt) if prompt else ''}-"
    # filename = f"{docs_hash}-{tokenizer_type}-{dictionary_type.replace('/', '_')}-{tokenizer_features}-{tokenizer_pos_filter}-{embedding_model.replace('/', '_')}-{representation_id}{nr_topics}"
    filename = f"{docs_hash}-{settings_hash}"
    path = Path(f"cache/{filename}")
    lock_path = Path(f"{path}.lock")
    logging.info(f"Lock not yet created at {lock_path}. Assuming first writer.")
    if not lock_path.exists():
        lock = FileLock(lock_path)
        logging.warning(f"Computation not started. Getting {lock}")
        with lock.acquire(poll_interval=1):
            if topic_model:
                logging.warning(f"Saving computed model to {path}")
                topics = topic_model.topics_
                probs = topic_model.probabilities_
                # topic_model.vectorizer_model = None # Prevent pickling of vectorizer
                topic_model.save(
                    path,
                    serialization="safetensors",
                    save_embedding_model=embedding_model,
                    save_ctfidf=True,
                )
                save_computed(
                    path,
                    topic_model.vectorizer_model,
                    topics,
                    probs,
                    embeddings,
                    reduced_embeddings,
                )
            else:
                logging.warning(f"Computing model {path}")
                (
                    topic_model,
                    embeddings,
                    reduced_embeddings,
                    topics,
                    probs,
                ) = calculate_model(
                    docs,
                    JapaneseTagger(
                        tokenizer_type,
                        dictionary_type,
                        ngram_range=ngram_range,
                        features=tokenizer_features,
                        pos_filter=tokenizer_pos_filter,
                        language=language,
                    ).vectorizer,
                    embedding_model,
                    representation_model,
                    prompt,
                    nr_topics,
                )
                logging.warning(f"Saving model {path}")
                # topic_model.vectorizer_model = None # Prevent pickling of vectorizer
                topic_model.save(
                    path,
                    serialization="safetensors",
                    save_embedding_model=embedding_model,
                    save_ctfidf=True,
                )
                save_computed(
                    path,
                    topic_model.vectorizer_model,
                    topics,
                    probs,
                    embeddings,
                    reduced_embeddings,
                )
    elif not topic_model:
        chech_finished = Path(f"{path}.finished")
        logging.warning(f"Loading precomputed model {path}")
        if chech_finished.exists():
            topic_model = BERTopic.load(str(path), embedding_model=embedding_model)
            # Currently not restored correctly:
            # https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_save_utils.py#L334
            # topic_model.vectorizer_model.tokenizer = tokenizer.tokenizer_fn
            # topic_model.vectorizer_model = vectorizer # restore vectorizer
            with open(f"{path}-vectorizer_model.pickle", "rb") as f:
                topic_model.vectorizer_model = cloudpickle.load(f)

            with open(f"{path}-topics.pickle", "rb") as f:
                topics = pickle.load(f)
            with open(f"{path}-probs.pickle", "rb") as f:
                probs = pickle.load(f)
            with open(f"{path}-embeddings.pickle", "rb") as f:
                embeddings = pickle.load(f)
            with open(f"{path}-reduced_embeddings.pickle", "rb") as f:
                reduced_embeddings = pickle.load(f)
        else:
            logging.error(
                f"Failed to access completed save (checked: {chech_finished}), sleeping for 1 second and retrying..."
            )
            sleep(1)
            return load_and_persist_model(
                docs,
                language,
                embedding_model,
                representation_model,
                prompt,
                nr_topics,
                tokenizer_type,
                dictionary_type,
                ngram_range,
                tokenizer_features,
                tokenizer_pos_filter,
                topic_model=topic_model,
                embeddings=embeddings,
                reduced_embeddings=reduced_embeddings,
                topics=topics,
                probs=probs,
            )
    else:  # Do nothing
        logging.warning("Noop: model already cached and loaded")
        if not topics:
            topics = topic_model.topics_
        if not probs:
            probs = topic_model.probabilites_
        assert topics is not None
        assert probs is not None
        assert embeddings is not None
        assert reduced_embeddings is not None

    logging.warning("Model loading complete.")

    return path, topic_model, embeddings, reduced_embeddings, topics, probs


(
    topic_model_path,
    topic_model,
    embeddings,
    reduced_embeddings,
    topics,
    probs,
) = load_and_persist_model(
    docs,
    language,
    st.session_state.embedding_model,
    st.session_state.representation_model,
    None,  # st.session_state.prompt,
    st.session_state.nr_topics,
    st.session_state.tokenizer_type,
    st.session_state.dictionary_type,
    st.session_state.ngram_range,
    st.session_state.tokenizer_features,
    st.session_state.tokenizer_pos_filter,
)


if topic_model.nr_topics != st.session_state.nr_topics:
    logging.warning(f"Reducing topics to {st.session_state.nr_topics}...")
    topic_model.reduce_topics(docs, st.session_state.nr_topics)
    (
        topic_model_path,
        topic_model,
        embeddings,
        reduced_embeddings,
        topics,
        probs,
    ) = load_and_persist_model(
        docs,
        language,
        st.session_state.embedding_model,
        st.session_state.representation_model,
        None,  # st.session_state.prompt,
        st.session_state.nr_topics,
        st.session_state.tokenizer_type,
        st.session_state.dictionary_type,
        st.session_state.ngram_range,
        st.session_state.tokenizer_features,
        st.session_state.tokenizer_pos_filter,
        topic_model=topic_model,
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
        # TODO Check these:
        topics=topic_model.topics_,
        probs=topic_model.probs_,
    )
elif (
    topic_model.nr_topics != st.session_state.nr_topics
    and topic_model.nr_topics != None
):
    # When setting topic reduction back off, we want to return to the previous state
    logging.warning(f"Loading model on nr_topics change: {st.session_state.nr_topics}")
    if st.session_state.nr_topics == None:
        st.session_state.nr_topics = "auto"
    (
        topic_model_path,
        topic_model,
        embeddings,
        reduced_embeddings,
        topics,
        probs,
    ) = load_and_persist_model(
        docs,
        language,
        st.session_state.embedding_model,
        st.session_state.representation_model,
        None,  # st.session_state.prompt,
        st.session_state.nr_topics,
        st.session_state.tokenizer_type,
        st.session_state.dictionary_type,
        st.session_state.ngram_range,
        st.session_state.tokenizer_features,
        st.session_state.tokenizer_pos_filter,
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
    doc_id = st.slider(
        "作品チャンク",
        min(work_docids),
        max(work_docids),
    )

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
            # FIXME Multi-aspect representations return a list of word, freqs that do not serialize with PyArrow; drop for now
        ]
    ]


with st.expander("Topic statistics"):
    st.write(get_topic_info(st.session_state.unique_id))
    # st.write(
    #     pl.DataFrame(
    #         (
    #             (aspect, topic_id, ", ".join(token for token, freq in token_freqs))
    #             for aspect, aspect_topics in topic_model.topic_aspects_.items()
    #             for topic_id, token_freqs in aspect_topics.items()
    #         ),
    #         schema=["Representation model", "Topic", "Tokens"],
    #     ).to_pandas()
    # )

# topic_name_to_id = {
#     r["Name"]: r["Topic"] for _, r in topic_model.get_topic_info().iterrows()
# }

# r_topic = st.selectbox("Choose topic", options=topic_name_to_id.keys())

# st.write(topic_model.get_representative_docs(topic=int(r_topic.split("_")[0])))


"# Topic information"

ttab1, ttab2, ttab3 = st.tabs(
    [
        "Hierarchical cluster analysis",
        "Similarity matrix heatmap",
        "Per-genre topic distribution",
    ]
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
    """Wrapper for token-level topic visualization. Currently, using the cf-tfifdf model is broken,
    so we use the embedding model instead."""
    _topic_distr, topic_token_distr = topic_model.approximate_distribution(
        doc,
        use_embedding_model=use_embedding_model,
        calculate_tokens=True,
        min_similarity=0.001,
        separator="",  # No whitespace for Japanese
    )

    # Visualize the token-level distributions
    df = topic_model.visualize_approximate_distribution(doc, topic_token_distr[0])
    assert not isinstance(df, pd.DataFrame) or not df.empty
    return df


use_embedding_model_option = st.checkbox("Use embedding model", False)

# https://www.aozora.gr.jp/cards/002231/files/62105_76819.html
# https://mysterytribune.com/suspense-novel-excerpt-the-echo-killing-by-christi-daugherty/
example_text = st.text_area(
    f"Token topic approximation using {'embedding' if use_embedding_model_option else 'c-tf-idf'} model",
    value="""金の羽根
昔あるところに、月にもお日さまにも増して美しい一人娘をお持ちの王さまとお妃さまがおりました。娘はたいそうおてんばで、宮殿中の物をひっくり返しては大騒ぎをしていました。"""
    if language == "Japanese"
    else """It was one of those nights.
Early on there was a flicker of hope— a couple of stabbings, a car wreck with potential. But the wounds weren’t serious and the accident was routine. After that it fell quiet.""",
)


st.write(
    visualize_text(st.session_state.unique_id, example_text, use_embedding_model_option)
)
