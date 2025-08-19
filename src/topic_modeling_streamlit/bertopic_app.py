import collections
import logging
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import litellm.exceptions
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
import torch
import xxhash
from bertopic import BERTopic
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from plotly.graph_objs._figure import Figure  # type: ignore[import-not-found]
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

from topic_modeling_streamlit.data_lib import create_corpus_core
from topic_modeling_streamlit.nlp_utils import (
    LanguageProcessor,
    apply_first_aspect_labels_core,
    ensure_aspect_core,
    generate_labels_core,
    get_default_prompt,
    get_embedding_model_options,
    llm_extract,
    llm_topic_suggestion,
    load_and_persist_model,
    model_fingerprint,
    stratified_weighted_document_sampling,
    topic_info_with_custom_names,
)
from topic_modeling_streamlit.plotting import (
    chunksizes_by_author_plot,
    visualize_documents_altair,
)
from topic_modeling_streamlit.presets import (
    LLM_REPRESENTATION_OPTIONS,
    NON_LLM_REPRESENTATION_OPTIONS,
    PRESETS,
)
from topic_modeling_streamlit.utils import (
    apply_ui_options,
    build_device_choices,
    filter_topic_choices,
    get_metadata_df,
    sample_representative_indices,
    strip_topic_prefix,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("filelock").setLevel(logging.DEBUG)


@retry(
    retry=retry_if_exception_type(litellm.exceptions.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=30),  # 1s → 2s → … → 30s
    stop=stop_after_delay(60),  # bail out after 60 s
)
def _retry_extract_topics(rep, model, docs_df, c_tf_idf, topics):
    """
    Wrap rep.extract_topics so that litellm RateLimitErrors back off + retry.
    """
    return rep.extract_topics(model, docs_df, c_tf_idf, topics)


@st.cache_data(show_spinner=True)
def cached_llm_extract(
    _cache_key: str,
    aspect: str,
    prompt: str,
    _model: BERTopic,
    docs_sample: list[str],
    tids: list[int],
) -> dict[int, list[tuple[str, float]]]:
    return llm_extract(
        aspect, prompt, _model, docs_sample, tids, st.session_state.device
    )


@st.cache_data(show_spinner=False)
def cached_llm_topic_suggestion(
    _cache_key: str,
    aspect: str,
    prompt: str,
    topic_id: int,
    _model: BERTopic,
    reps_docs: list[str],
) -> str:
    return llm_topic_suggestion(
        aspect, prompt, topic_id, _model, reps_docs, st.session_state.device
    )


def set_options(reload: bool) -> None:
    """Update session state from UI options and optionally mark for reload."""
    if not reload:
        st.session_state["computed"] = False
    apply_ui_options(st.session_state, reload=reload)
    st.toast("Compute!", icon="🎉")


st.set_page_config(layout="wide")


st.title("BERTopic Playground")

# Seed applied defaults; form *_option keys control pending values
st.session_state.setdefault(
    "applied_representation_model",
    st.session_state.get("representation_model_option", ["KeyBERTInspired"]),
)
default_prompt_tpl = get_default_prompt(
    st.session_state.get("language_option", "Japanese")
)
st.session_state.setdefault(
    "applied_prompt", st.session_state.get("prompt_option", default_prompt_tpl)
)
st.markdown(
    """
-   Application code: <https://github.com/borh/TopicModelingStreamlit>
-   BERTopic Github: <https://github.com/MaartenGr/BERTopic>
-   Official documentation: <https://maartengr.github.io/BERTopic/index.html>
"""
)

st.sidebar.header("BERTopic Settings")

# Presets (filtered by selected language)
with st.sidebar:
    st.markdown("### Presets")
    curr_lang = st.session_state.get("language_option", "Japanese")
    lang_presets = [
        name for name, cfg in PRESETS.items() if cfg.get("language_option") == curr_lang
    ]
    preset_name = st.selectbox(
        "Load recommended options",
        options=["None"] + lang_presets,
        index=0,
        key="preset_selector",
        help="Selecting a preset fills the sidebar options. Click Compute! to run.",
    )
    if preset_name and preset_name != "None":
        cfg = PRESETS[preset_name]
        for k, v in cfg.items():
            st.session_state[k] = v
        # Any preset update changes corpus-affecting params → force model reload
        st.session_state["reload"] = True
        # Leave settings in *_option keys; Compute! will apply them
        st.info(f"Preset applied: {preset_name}. Review options then click Compute!")


language = st.sidebar.radio(
    "Language",
    ("Japanese", "English"),
    index=0,
    horizontal=True,
    key="language_option",
    on_change=set_options,
    args=(False,),
)


@st.cache_data(show_spinner=True)
def generate_labels(
    _unique_id: str,
    nr_words: int,
    aspect: str | None,
    _model: BERTopic,
    docs: list[str],
) -> list[str]:
    """
    Delegate to core labeling logic without Streamlit dependencies.
    """

    prompt_val = st.session_state.get(
        "applied_prompt", st.session_state.get("prompt", "")
    )
    return generate_labels_core(
        _model, nr_words, aspect, prompt_val, st.session_state.device, docs
    )


def apply_first_aspect_labels(topic_model: BERTopic, docs: list[str]) -> None:
    """
    Update topic_model.custom_labels_ using core logic.
    """

    first_aspect = (
        st.session_state["applied_representation_model"][0]
        if st.session_state.get("applied_representation_model")
        else None
    )
    if not first_aspect:
        return
    nr_words = (
        1
        if first_aspect in LLM_REPRESENTATION_OPTIONS
        else st.session_state.get("nr_label_words", 3)
    )
    try:
        prompt_val = st.session_state.get(
            "applied_prompt", st.session_state.get("prompt", "")
        )
        apply_first_aspect_labels_core(
            topic_model, first_aspect, nr_words, prompt_val, st.session_state.device
        )
    except RetryError:
        st.warning("Rate limited while updating labels; keeping previous labels.")
        return


def ensure_aspect(topic_model: BERTopic, aspect: str, docs: list[str]) -> None:
    """
    Delegate aspect computation to core function.
    """

    prompt_val = st.session_state.get(
        "applied_prompt", st.session_state.get("prompt", "")
    )
    # Cheap short‑circuit for LLM aspects: if we already have entries for all non‑outlier topics, skip
    if aspect in LLM_REPRESENTATION_OPTIONS:
        info = topic_model.get_topic_info().reset_index(drop=True)
        tids = [int(t) for t in info["Topic"].tolist() if int(t) != -1]
        amap = topic_model.topic_aspects_.get(aspect, {})
        if amap and all(t in amap and amap[t] for t in tids):
            return
    ensure_aspect_core(
        topic_model,
        aspect,
        prompt_val,
        st.session_state["language"],
        docs,
        st.session_state.device,
    )
    return


def rep_settings_fragment() -> None:
    """
    Render representation and prompt controls without callbacks.
    Values are applied when the main 'Compute!' form is submitted.
    """
    lang = st.session_state.get("language_option", "Japanese")
    opts = NON_LLM_REPRESENTATION_OPTIONS + LLM_REPRESENTATION_OPTIONS
    # Multiselect for representation set. Do not mutate applied_* here.
    # Seed the form key on first load so the UI shows a sensible default
    st.session_state.setdefault(
        "representation_model_option",
        st.session_state.get("applied_representation_model", ["KeyBERTInspired"]),
    )
    st.multiselect(
        "Topic representation",
        opts,
        key="representation_model_option",
        help="Select one or more methods to generate topic labels/representations.",
    )
    # Prompt editor (collapsible), no side effects; value stored in prompt_option
    default_prompt = get_default_prompt(lang)
    st.session_state.setdefault(
        "prompt_option", st.session_state.get("applied_prompt", default_prompt)
    )
    with st.expander("Prompt", expanded=False):
        st.text_area(
            "Prompt",
            key="prompt_option",
            label_visibility="collapsed",
            help="Custom prompt for topic labeling. Use [DOCUMENTS] and [KEYWORDS] as placeholders.",
        )


def create_settings_form() -> st.form:
    """Create the settings form in the sidebar, with clearer section grouping."""
    # fall back to Japanese if this hasn’t been initialized yet
    lang = st.session_state.get("language_option", "Japanese")

    settings = st.sidebar.form("settings")

    with settings:
        # Adjust settings and press Compute! to run.
        st.caption("Adjust settings and press Compute! to run.")

        st.markdown("### Topic Model Settings")
        settings.selectbox(
            "Embedding model",
            get_embedding_model_options(lang),
            key="embedding_model_option",
            help="""Choose the sentence-embedding model used to vectorize your documents.
    
**Key considerations:**
- Models trained on target language data (e.g., Japanese models for Japanese text) typically perform better
- Larger models generally produce better embeddings but require more compute/memory
- Multilingual models offer versatility but may sacrifice some language-specific performance
- JMTEB/MTEB leaderboard scores can guide selection but real-world performance varies by domain

The embedding model transforms each document into a dense vector representation that captures semantic meaning.""",
        )
        # Topic representation controls live inside the form; applied on Compute!
        st.markdown("#### Topic representation")
        rep_settings_fragment()

        # Resume the form for the rest of the settings
        # Number of words to use in each topic label (non-LLM representations only)
        settings.number_input(
            "Label word count",
            min_value=1,
            max_value=10,
            value=3,
            key="nr_label_words_option",
            help="Number of words for each generated topic label (non-LLM reps). LLM-based labels always use 1 word.",
        )

        # If any "/" model was chosen, allow a custom prompt
        settings.number_input(
            "Reduce to n topics (0 = auto)",
            min_value=0,
            value=0,
            step=1,
            key="nr_topics_option",
            help="""Fix the number of topics (0 = let BERTopic decide automatically).

**Topic reduction strategies:**
- **0 (auto)**: Uses HDBSCAN's natural cluster detection - recommended for exploratory analysis
- **Fixed number**: Forces model to merge similar topics until target is reached
- Topics are merged based on c-TF-IDF similarity, preserving the most distinctive topics
- Reduction happens after initial clustering, so setting this too low may merge meaningful distinctions""",
        )
        device_choices = build_device_choices()
        settings.radio(
            "Hardware",
            device_choices,
            key="device_option",
            help="Select compute device: CPU (universal), CUDA (NVIDIA GPUs), or MPS (Apple Silicon M-series).",
        )

        st.divider()

        st.markdown("### Tokenizer Settings")
        if lang == "Japanese":
            tokenizer_dictionary_options = {
                "MeCab": ["近現代口語小説UniDic", "UniDic-CWJ", "UniDic-CSJ"],
                "Sudachi": [
                    "SudachiDict-full/A",
                    "SudachiDict-full/B",
                    "SudachiDict-full/C",
                ],
                "spaCy": ["ja_core_news_sm"],
                "Juman++": ["Jumandict"],
            }
        else:
            tokenizer_dictionary_options = {
                "spaCy": ["en_core_web_sm", "en_core_web_trf"],
            }
        settings.selectbox(
            "Tokenizer / Dictionary",
            [
                f"{tokenizer}/{dictionary}"
                for tokenizer, dictionaries in tokenizer_dictionary_options.items()
                for dictionary in dictionaries
            ],
            key="tokenizer_dictionary_option",
            help="""Choose tokenizer engine and dictionary for text segmentation.

**Japanese tokenizers:**
- **MeCab + UniDic**: Excellent for modern/contemporary texts, especially fiction
- **Sudachi**: Highly customizable with A/B/C split modes for different granularities (A is euqivalent to Short Unit Words, and C similar to Long Unit Words in UniDic)
- **spaCy**: Mostly equivalent with Sudachi, but using a smaller dictionary
- **Juman++**: Different POS hierarchy and token sizes ("である" is one token)

**Dictionary selection impacts:**
- Vocabulary coverage (OOV handling)
- Morphological analysis accuracy
- Domain-specific terminology recognition""",
        )
        settings.multiselect(
            "Token features",
            ["orth", "lemma", "pos1"],
            ["orth"],
            key="tokenizer_features_option",
            help="""Which attributes to extract per token.

**Feature types:**
- **orth**: Surface form (exact text as it appears)
- **lemma**: Dictionary/base form (e.g., 食べた → 食べる)
- **pos1**: Part-of-speech tag (noun, verb, etc.)

**Combinations:**
- Multiple features are concatenated with '/' (e.g., "食べる/動詞")
- Lemmatization helps group inflected forms but may lose nuance
- POS tags can help distinguish homographs but increase vocabulary size""",
        )
        # Decide which POS tags can be removed
        if lang == "Japanese":
            dict_type = st.session_state.get("dictionary_type", "")
            if dict_type == "Jumandict":
                # Juman++ specific tags
                pos_choices = [
                    "名詞",
                    "指示詞",
                    "動詞",
                    "形容詞",
                    "判定詞",
                    "助動詞",
                    "副詞",
                    "助詞",
                    "接続詞",
                    "連体詞",
                    "感動詞",
                    "接頭辞",
                    "接尾辞",
                    "特殊",
                    "未定義語",
                ]
            else:
                # Default Japanese POS tags
                pos_choices = [
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
        else:
            # English POS tags
            pos_choices = [
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
        settings.multiselect(
            "Remove POS tags",
            pos_choices,
            [],
            key="tokenizer_pos_filter_option",
            help="""Exclude tokens matching these part-of-speech tags.

**Common filtering strategies:**

- Remove symbols (記号) and whitespace (空白) to reduce noise
- Remove particles (助詞) and auxiliary verbs (助動詞) for content-focused analysis
- Keep only nouns (名詞) and verbs (動詞) for topic modeling
- Filter affects both vocabulary size and topic interpretability

**Note**: Over-filtering may remove important contextual information.""",
        )
        # Ensure the session state key is initialized before creating the widget
        if "surface_filter_option" not in st.session_state:
            st.session_state["surface_filter_option"] = ""

        settings.text_input(
            "Filter tokens (regex)",
            key="surface_filter_option",
            help="Enter a regex pattern; any token matching it will be excluded. Use `(a|b)` etc. to specify multiple patterns.\nIn order to strip all punctuation character tokens, you can try: `^[\\u3000-\\u303f\\uff00-\\uffef\\p{P}]+$`",
        )
        settings.select_slider(
            "N-gram range",
            options=range(1, 5),
            value=(1, 1),
            key="ngram_range_option",
            help="""Min and max n-gram length for feature extraction.

**N-gram considerations:**
- **(1,1)**: Unigrams only - captures individual words/tokens
- **(1,2)**: Unigrams + bigrams - captures common phrases
- **(2,2)**: Bigrams only - focuses on word pairs
- Higher n-grams capture longer phrases but exponentially increase vocabulary

**Trade-offs:**
- Larger n-grams better capture multi-word concepts (e.g., "machine learning")
- But increase sparsity and computational cost
- Japanese often benefits from bigrams due to compound words, but using the SudachiDict-full/C dictionary is often preferrable""",
        )
        settings.number_input(
            "Max document frequency",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            key="max_df_option",
            help="Ignore tokens that appear in more than this fraction of documents.",
        )

        st.divider()

        st.markdown("### Corpus Settings")
        settings.number_input(
            "Chunk size (tokens)",
            min_value=10,
            value=100,
            step=10,
            key="chunksize_option",
            help="""Number of tokens per text chunk.

**Chunking considerations:**
- **Small chunks (50-100)**: Better topic granularity, more documents for clustering
- **Large chunks (200-500)**: More context per document, fewer but richer documents
- **Very large (500+)**: Risk mixing multiple topics in one document

**Note**: The chunks size should not be greater than what the selected SentenceTransformer model can support. For many models, this used to be 512 (subword) tokens, but modern models often support up to 8k tokens.

**Domain-specific guidance:**
- Fiction/narrative: 100-200 tokens captures scene-level topics
- Academic/technical: 200-300 tokens preserves argument structure
- Social media/short texts: May not need chunking

Chunks respect sentence boundaries when possible.""",
        )
        settings.number_input(
            "Min chunk size (tokens)",
            min_value=1,
            value=20,
            max_value=st.session_state.get("chunksize_option", 100),
            step=1,
            key="min_chunksize_option",
            help="Drop any chunk smaller than this number of tokens.",
        )
        settings.number_input(
            "Chunks per document (0 = all)",
            min_value=0,
            value=50,
            step=1,
            key="chunks_option",
            help="Limit how many chunks to extract per document (0 = take all).",
        )

    return settings


settings = create_settings_form()
submit_all = settings.form_submit_button("Compute!", on_click=set_options, args=(True,))

# Initialize the persistent flag
if "computed" not in st.session_state:
    st.session_state["computed"] = False

# When the Compute! button is pressed, mark that we've computed
if submit_all:
    st.session_state["computed"] = True

# Ensure unique_id is set on first load
if "unique_id" not in st.session_state:
    set_options(False)

# Ensure device bare key is always in sync in case set_options wasn't called
if "device_option" in st.session_state:
    st.session_state["device"] = st.session_state["device_option"]

# Gate all heavy work until after the first Compute!
if not st.session_state["computed"]:
    st.info(
        "Adjust your language/model/tokenizer options in the sidebar and then click **Compute!**"
    )
    st.stop()

st.sidebar.markdown(
    f"""
### References
-   <https://github.com/MaartenGr/BERTopic>
-   <https://arxiv.org/abs/2203.05794>
-   <https://github.com/borh/TopicModelingStreamlit>

Running on {socket.gethostname()}
"""
)


all_metadata = get_metadata_df(st.session_state["language"], ui_warn=True)

# Ensure current UI options (sidebar *_option values) are applied before corpus creation:
apply_ui_options(st.session_state, reload=False)

if st.session_state["language"] == "Japanese":
    text_dir = Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/")
    if not text_dir.exists():
        st.error(
            "Missing Aozora texts directory: Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/\n"
            "Please download the text files into this folder."
        )
else:
    text_dir = Path("./standard-ebooks-selection/")
    if not text_dir.exists():
        st.error(
            "Missing Standard Ebooks texts directory: standard-ebooks-selection/\n"
            "Please place the .txt files into this folder."
        )


@st.cache_data(show_spinner=True, show_time=True)
def create_corpus(
    _all_metadata: pl.DataFrame | None,
    language: str,
    chunksize: int,
    min_chunksize: int,
    chunks: int,
    tokenizer_type: str,
    dictionary_type: str,
) -> tuple[list[str], pl.DataFrame]:
    """
    Cached wrapper around data_lib.create_corpus_core. Keeps Streamlit cache and UI intact.
    """

    return create_corpus_core(
        _all_metadata,
        language,
        chunksize,
        min_chunksize,
        chunks,
        tokenizer_type,
        dictionary_type,
        tokenizer_factory=lambda **kw: LanguageProcessor(**kw),
    )


if all_metadata is None:
    st.stop()
docs, metadata = create_corpus(
    all_metadata,
    st.session_state["language"],
    st.session_state["chunksize"],
    st.session_state["min_chunksize"],
    st.session_state["chunks"],
    st.session_state["tokenizer_type"],
    st.session_state["dictionary_type"],
)
st.session_state["docs_full"] = list(docs)
st.session_state["metadata_full"] = metadata.clone()
assert len(st.session_state["docs_full"]) == st.session_state["metadata_full"].height, (
    f"docs/metadata length mismatch right after corpus creation: "
    f"{len(st.session_state['docs_full'])} vs {st.session_state['metadata_full'].height}"
)

# Clamp doc_id globally to valid range for docs/metadata
if "doc_id" in st.session_state:
    max_idx = len(docs) - 1
    if st.session_state["doc_id"] > max_idx:
        st.session_state["doc_id"] = max_idx


authors = list(
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
    -   Works: {metadata.select(pl.col("title")).n_unique()}
    -   Docs: {len(docs)}
        - Sample: {docs[0][:100]}
    -   Chunksize: {st.session_state.chunksize}
    -   Chunks/doc: {st.session_state.chunks}
    -   Topic reduction (limit): {st.session_state.nr_topics}
    -   Model: {st.session_state.embedding_model}
    -   Representation: {st.session_state.applied_representation_model}
    -   Active labeling aspect: {(st.session_state["applied_representation_model"][0] if st.session_state["applied_representation_model"] else "None")}
    """
    )


if not metadata.is_empty():
    with st.expander("Open to see basic document stats"):
        st.write(chunksizes_by_author_plot(metadata))
        # Author distribution
        st.write(
            px.histogram(
                x=metadata.get_column("author").to_list(),
                title="Author document distribution",
                labels={"x": "author"},
            )
        )
        # Genre length distribution
        st.write(
            px.box(
                x=metadata.get_column("genre").to_list(),
                y=metadata.get_column("length").to_list(),
                hover_data={
                    "label": metadata.get_column("label").to_list(),
                    "docid": metadata.get_column("docid").to_list(),
                },
                title="Genre document length (characters) distribution",
                labels={"x": "genre", "y": "characters"},
            )
        )
        # Genre counts
        st.write(
            px.histogram(
                x=metadata.get_column("genre").to_list(),
                title="Genre document distribution",
                labels={"x": "genre"},
            )
        )

# Load/fit model only once per Compute!
# If we've never loaded it, or options changed (`reload`), call load_and_persist_model.
if st.session_state.get("reload") or "topic_model" not in st.session_state:
    # Apply all current UI options and mark for reload
    apply_ui_options(st.session_state, reload=True)

    try:
        topic_model_path, topic_model, embeddings, reduced_embeddings, topics, probs = (
            load_and_persist_model(
                docs,
                st.session_state["language"],
                st.session_state.embedding_model,
                st.session_state["applied_representation_model"],
                st.session_state["applied_prompt"],
                st.session_state.nr_topics,
                st.session_state.tokenizer_type,
                st.session_state.dictionary_type,
                st.session_state.ngram_range,
                st.session_state.max_df,
                st.session_state.tokenizer_features,
                st.session_state.tokenizer_pos_filter,
                surface_filter=st.session_state.surface_filter,
                device=st.session_state.device,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                },
            )
        )
    except RetryError:
        st.error(
            "Rate limit exceeded for the selected LLM provider. "
            "Please pick another model or wait a moment and retry."
        )
        st.stop()
    except ValueError as e:
        # Surface tokenization / empty-corpus errors to the UI with a clear message.
        st.error(f"Unable to build model: {e}")
        st.stop()
    st.session_state.update(
        {
            "topic_model_path": topic_model_path,
            "topic_model": topic_model,
            "embeddings": embeddings,
            "reduced_embeddings": reduced_embeddings,
            "topics": topics,
            "probs": probs,
        }
    )
    st.session_state["reload"] = False
    first_aspect = (
        st.session_state["applied_representation_model"][0]
        if st.session_state["applied_representation_model"]
        else None
    )
    if first_aspect and first_aspect in LLM_REPRESENTATION_OPTIONS:
        with st.spinner(f"Generating default labels with {first_aspect}..."):
            apply_first_aspect_labels(topic_model, docs)
    else:
        apply_first_aspect_labels(topic_model, docs)
    # Populate/refresh all selected aspect representations without refit
    for asp in st.session_state.get("applied_representation_model", []):
        try:
            if asp in LLM_REPRESENTATION_OPTIONS:
                with st.spinner(f"Computing LLM representation: {asp}..."):
                    ensure_aspect(topic_model, asp, docs)
            else:
                with st.spinner(f"Computing representation: {asp}"):
                    ensure_aspect(topic_model, asp, docs)
        except RateLimitError as e:
            st.warning(f"Skipping aspect {asp} due to LLM rate limit: {e}")
            continue
        except ServiceUnavailableError as e:
            st.warning(f"Skipping aspect {asp} due to LLM service unavailability: {e}")
            continue
        except Exception as e:
            st.warning(f"Skipping aspect {asp} due to error: {e}")
            continue
else:
    topic_model_path = st.session_state["topic_model_path"]
    topic_model = st.session_state["topic_model"]
    embeddings = st.session_state["embeddings"]
    reduced_embeddings = st.session_state["reduced_embeddings"]
    topics = st.session_state["topics"]
    probs = st.session_state["probs"]


# Initialize default custom labels only once (so subsequent reruns or user edits aren't clobbered)
if not getattr(topic_model, "custom_labels_", None):
    first_aspect = (
        st.session_state["applied_representation_model"][0]
        if st.session_state["applied_representation_model"]
        else None
    )
    if first_aspect is not None:
        label_words = (
            1
            if first_aspect in LLM_REPRESENTATION_OPTIONS
            else st.session_state.get("nr_label_words", 3)
        )
        try:
            if first_aspect in LLM_REPRESENTATION_OPTIONS:
                with st.spinner(f"Generating default labels with {first_aspect}..."):
                    default_labels = generate_labels(
                        st.session_state.unique_id,
                        label_words,
                        first_aspect,
                        topic_model,
                        docs,
                    )
            else:
                default_labels = generate_labels(
                    st.session_state.unique_id,
                    label_words,
                    first_aspect,
                    topic_model,
                    docs,
                )
        except RetryError:
            st.error(
                f"Rate limit exceeded for LLM provider {first_aspect}. "
                "Please try another model or wait a moment and retry."
            )
            st.stop()
        # store them both in the model and on the custom_labels_ attribute
        topic_model.custom_labels_ = default_labels
        topic_model.set_topic_labels(default_labels)

if topic_model.nr_topics != st.session_state.nr_topics:
    logging.warning(f"Reducing topics to {st.session_state.nr_topics}...")
    topic_model.reduce_topics(docs, st.session_state.nr_topics)
    (topic_model_path, topic_model, embeddings, reduced_embeddings, topics, probs) = (
        load_and_persist_model(
            docs,
            st.session_state["language"],
            st.session_state.embedding_model,
            st.session_state["applied_representation_model"],
            st.session_state["applied_prompt"],
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
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
        )
    )
    apply_first_aspect_labels(topic_model, docs)
elif (
    topic_model.nr_topics != st.session_state.nr_topics
    and topic_model.nr_topics is not None
):
    logging.warning(f"Loading model on nr_topics change: {st.session_state.nr_topics}")
    if st.session_state.nr_topics is None:
        st.session_state.nr_topics = "auto"

    # After a topics change, also refresh state
    st.session_state["reload"] = True

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


if not st.session_state.applied_representation_model:
    st.markdown("## Topic word browser")

    c01, c02 = st.columns(2)

    with c01:
        top_n_topics = st.number_input("Top n topics", min_value=1, value=10)
    with c02:
        n_words = st.number_input("n words", min_value=1, value=8)

    st.write(
        topic_model.visualize_barchart(
            top_n_topics=top_n_topics,
            n_words=n_words,
            custom_labels=True,
        )
    )
    with st.expander("Explanation"):
        st.markdown(
            """
            ### Topic Word Browser
            
            This visualization presents the most important words for each topic based on **c-TF-IDF scores** (class-based Term Frequency-Inverse Document Frequency).
            
            **How to interpret:**
            - **Bar length** indicates the word's importance to that topic (c-TF-IDF score)
            - **Words** are ranked by how distinctive they are to each topic vs. other topics
            - Topics are labeled with either auto-generated IDs or custom labels if representation models were used
            
            **c-TF-IDF scoring:**
            - Combines frequency of word in topic (TF) with its uniqueness across topics (IDF)
            - High scores = words that are frequent in this topic but rare in others
            - This identifies the most discriminative vocabulary for each topic
            
            **Analysis tips:**
            - Look for semantic coherence within each topic's top words
            - Check if topics are too similar (overlapping vocabulary) - may indicate need for topic reduction
            - Very generic words across all topics suggest need for better preprocessing or stop word removal
            """
        )


"## Document and topic 2D plot (UMAP)"


@st.cache_data(show_spinner=True, show_time=True)
def visualize_docs(
    _unique_id: str,
    color_by: str,
    docs: list[str],
    embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    selectable: bool = False,
    show_outliers: bool = True,
) -> alt.Chart:
    """
    2D scatter of documents. We pass both the full embeddings and
    their 2D UMAP reduction so BERTopic can color each point by topic.
    unique_id is the only thing that determines cache invalidation.
    """
    # Build an explicit label_map from the current topic model so labels are keyed by topic id
    topic_info = topic_model.get_topic_info().reset_index(drop=True)
    label_map = {
        int(r["Topic"]): (r.get("CustomName") or r.get("Name") or "")
        for r in topic_info.to_dict(orient="records")
    }

    chart = visualize_documents_altair(
        docs=docs,
        topic_model=topic_model,
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
        sample=1.0,
        custom_labels=True,
        hide_annotations=False,
        width=800,
        height=600,
        authors=metadata.get_column("author").to_list(),
        works=metadata.get_column("title").to_list(),
        chunk_ids=metadata.get_column("docid").to_list(),
        color_by=color_by,
        selectable=selectable,
        show_outliers=show_outliers,
        topics_per_doc=topics,  # explicit per-document topics array from the caller scope
        label_map=label_map,
        device=st.session_state.get("device", "cpu"),
    )
    return chart


st.checkbox(
    "Show outlier topic -1?",
    value=False,
    key="show_outliers_option",
    help="Outliers are documents assigned to topic -1 by HDBSCAN. Hide to declutter plots.",
)
color_by = st.radio(
    "Color points by",
    ["Topic", "Author", "Work"],
    index=0,
    horizontal=True,
)

# Document and topic 2D plot (UMAP)
# Include custom labels in cache key to update when labels change
labels_hash = xxhash.xxh3_64_hexdigest(
    str(getattr(topic_model, "custom_labels_", [])).encode()
)
cache_key = f"{st.session_state.unique_id}_{labels_hash}_{st.session_state.get('labels_version', 0)}"

try:
    st.altair_chart(
        visualize_docs(
            cache_key,
            color_by,
            docs,
            embeddings,
            reduced_embeddings,
            # selectable=False # This is not yet supported by Streamlit
            show_outliers=bool(st.session_state.get("show_outliers_option", False)),
        ),
        theme="streamlit",
        use_container_width=True,
        key="umap_plot",
    )
except Exception as e:
    st.error(f"Error rendering document/topic plot: {e}")
with st.expander("Explanation"):
    st.markdown(
        """
        ### Document and Topic 2D Plot (UMAP)
        
        This scatter plot shows all document chunks projected into 2D space using **UMAP** (Uniform Manifold Approximation and Projection) dimensionality reduction.
        
        **Understanding the visualization:**
        - **Position**: Documents close together have similar content/embeddings
        - **Colors**: Indicate topic assignment (or author/work based on selection)
        - **Clusters**: Dense regions represent coherent topics; scattered points may be outliers
        - **Topic labels**: Positioned at cluster centroids
        
        **UMAP algorithm properties:**
        - Preserves both local and global structure of high-dimensional data
        - Distance between clusters is meaningful (unlike t-SNE)
        - Parameters: n_neighbors=10 (local vs global structure), min_dist=0.0 (cluster tightness)
        
        **Analysis guidance:**
        - Well-separated clusters indicate distinct topics
        - Overlapping regions suggest related or transitional content
        - Outliers (isolated points) may represent unique documents or noise
        - Color by different attributes to discover patterns (topic structure vs. authorship vs. work)
        
        **Interactive features:**
        - Hover over points to see document content and metadata
        - Zoom and pan to explore dense regions
        - Use color-by options to reveal different document relationships
        """
    )

"## Corpus topic browser"


@st.cache_data(show_spinner=True)
def find_topics_cached(
    _unique_id: str, _model: BERTopic, query: str
) -> tuple[list[int], list[float]]:
    return _model.find_topics(query)


@st.cache_data(show_spinner=True)
def get_author_works(metadata: pl.DataFrame, author: str) -> list[str]:
    return (
        metadata.filter(pl.col("author") == author)
        .select(pl.col("title"))
        .unique()
        .sort(pl.col("title"))
        .get_column("title")
        .to_list()
    )


selection_col1, selection_col2 = st.columns(2)


def set_navigation_to_doc(docid: int):
    """
    Centralized navigation update: sets doc_id, slider_doc_id, input_doc_id,
    and updates author, work, last_work from metadata for the given docid.
    """
    docid = min(max(0, docid), len(docs) - 1)
    st.session_state["doc_id"] = docid
    st.session_state["slider_doc_id"] = docid
    st.session_state["input_doc_id"] = docid
    sel = metadata.filter(pl.col("docid") == docid)
    if not sel.is_empty():
        st.session_state["author"] = sel["author"][0]
        st.session_state["work"] = sel["title"][0]
        st.session_state["last_work"] = st.session_state["work"]


def _update_for_work():
    selected_work_now = st.session_state.get("work")
    work_docids_local = (
        st.session_state["metadata_full"]
        .filter(pl.col("title") == selected_work_now)
        .select(pl.col("docid"))
        .to_series()
        .to_list()
    )
    if work_docids_local:
        set_navigation_to_doc(min(work_docids_local))


def _on_author_change():
    new_author = st.session_state.get("author")
    works_for_author = get_author_works(st.session_state["metadata_full"], new_author)
    if works_for_author:
        if st.session_state.get("work") not in works_for_author:
            st.session_state["work"] = works_for_author[0]
    _update_for_work()


with selection_col1:
    selected_author = st.selectbox(
        "Author",
        options=authors,
        key="author",
        on_change=_on_author_change,
    )

author_works = get_author_works(
    st.session_state["metadata_full"], st.session_state.get("author")
)
if st.session_state.get("work") not in author_works:
    st.session_state["work"] = author_works[0]

with selection_col2:
    selected_work = st.selectbox(
        "Work",
        options=author_works or [],
        key="work",
        on_change=_update_for_work,
    )

work_docids = (
    st.session_state["metadata_full"]
    .filter(pl.col("title") == selected_work)
    .select(pl.col("docid"))
    .to_series()
    .to_list()
)


if work_docids:
    lo, hi = min(work_docids), max(work_docids)

    st.session_state.setdefault("doc_id", lo)
    st.session_state.setdefault("slider_doc_id", lo)
    st.session_state.setdefault("input_doc_id", lo)

    def _on_slider_change():
        v = st.session_state["slider_doc_id"]
        set_navigation_to_doc(v)
        print(
            f"[slider-change] slider_doc_id={v} doc_id={st.session_state.get('doc_id')} "
            f"input_doc_id={st.session_state.get('input_doc_id')} author={st.session_state.get('author')!r} "
            f"work={st.session_state.get('work')!r} last_work={st.session_state.get('last_work')!r}"
        )

    def _on_input_change():
        v = st.session_state["input_doc_id"]
        set_navigation_to_doc(v)
        print(
            f"[input-change] input_doc_id={v} doc_id={st.session_state.get('doc_id')} "
            f"slider_doc_id={st.session_state.get('slider_doc_id')} author={st.session_state.get('author')!r} "
            f"work={st.session_state.get('work')!r} last_work={st.session_state.get('last_work')!r}"
        )

    col_slider, col_input = st.columns([3, 1])
    with col_slider:
        st.slider(
            "Work chunk",
            min_value=lo,
            max_value=hi,
            key="slider_doc_id",
            on_change=_on_slider_change,
        )
    with col_input:
        st.number_input(
            "Go to chunk id",
            min_value=0,
            max_value=len(docs) - 1,
            key="input_doc_id",
            on_change=_on_input_change,
            help="Navigate to any chunk ID in the entire corpus. Author and work will update automatically.",
        )

    # Clamp current doc_id to within bounds for docs list
    doc_id = st.session_state.get("doc_id", 0) or 0
    if doc_id >= len(docs):
        doc_id = max(0, len(docs) - 1)
        st.session_state["doc_id"] = doc_id
    st.markdown(f"> {docs[doc_id]}")
    st.write(topic_model.visualize_distribution(probs[doc_id], min_probability=0.0))


st.markdown("## Topic Representations & Interpretation")


def _sync_selected_topic(changed_key: str):
    """
    Whenever one tab’s selectbox changes, write its value into
    st.session_state['selected_topic'] and propagate to the other two.
    """
    sel = st.session_state[changed_key]
    st.session_state["selected_topic"] = sel
    for k in ("selected_topic_docs", "selected_topic_label"):
        if k != changed_key:
            st.session_state[k] = sel


tab_names = ["Topic Overview", "Representative Documents", "Label Management"]

default_tab = st.session_state.get("selected_tab_name")
if default_tab not in tab_names:
    default_tab = "Topic Overview"

# Single radio definition
selected_tab = st.radio(
    "Select view",
    options=tab_names,
    index=tab_names.index(default_tab),
    key="selected_tab_name",
    horizontal=True,
)

# build the `full_topic_info` exactly as before
base_info_df = topic_model.get_topic_info().reset_index(drop=True)
topic_info_df = topic_info_with_custom_names(
    base_info_df, getattr(topic_model, "custom_labels_", None)
)
rep_df = pd.json_normalize(topic_info_df["Representation"].tolist())
rep_df = rep_df.loc[:, rep_df.applymap(lambda x: bool(x)).any()]
full_topic_info = pd.concat(
    [topic_info_df.drop(columns=["Representation", "Representative_Docs"]), rep_df],
    axis=1,
)

# build the topic selector options once
topic_choices: list[str] = []
topic_idx_map: dict[int, int] = {}

records = topic_info_df.to_dict(orient="records")
# Collect all topic ids and names first
tids = [int(r["Topic"]) for r in records]
names = [r.get("CustomName") or r.get("Name") or "" for r in records]
clean_names = strip_topic_prefix(names, tids)

for i, tid0 in enumerate(tids):
    topic_choices.append(f"{tid0}: {clean_names[i]}")
    topic_idx_map[tid0] = i
# Compute visible choices based on the checkbox but keep full choices for indexing
show_outliers = bool(st.session_state.get("show_outliers_option", False))

visible_choices = filter_topic_choices(topic_choices, show_outliers)


# Ensure a valid selected value if previously set to hidden -1
def _coerce_visible(key: str):
    v = st.session_state.get(key)
    if v and v not in visible_choices and visible_choices:
        st.session_state[key] = visible_choices[0]


if selected_tab == "Topic Overview":
    st.caption("Overview of your topics and default labels")
    # Strip out any empty entries in list-columns; skip non-list cells
    for col in full_topic_info.columns.difference(
        ["Topic", "Count", "Name", "CustomName"]
    ):
        full_topic_info[col] = full_topic_info[col].apply(
            lambda items: [x for x in items if x] if isinstance(items, list) else items
        )
    if not show_outliers:
        full_topic_info = full_topic_info[full_topic_info["Topic"] != -1]
    st.dataframe(full_topic_info, use_container_width=True)

elif selected_tab == "Representative Documents":
    _coerce_visible("selected_topic_docs")
    st.selectbox(
        "Select topic to analyze",
        visible_choices,
        key="selected_topic_docs",
        on_change=_sync_selected_topic,
        args=("selected_topic_docs",),
    )
    tid = int(st.session_state["selected_topic_docs"].split(":", 1)[0])
    st.subheader(f"Representative Documents for Topic {tid}")
    n_docs = int(
        st.number_input(
            "Number of representative docs",
            min_value=1,
            max_value=20,
            value=5,
            key=f"repr_docs_n_{tid}",
        )
    )
    docs_all = st.session_state["docs_full"]
    metadata_all = st.session_state["metadata_full"]
    topic_indices = [i for i, t in enumerate(topic_model.topics_) if t == tid]
    reps_idx = []
    if topic_indices:
        sampled_map = stratified_weighted_document_sampling(
            docs_all,
            metadata_all,
            topic_model.topics_,
            probs,
            nr_docs=n_docs,
            stratify_by="genre",
            seed=42,
        )
        reps_texts = sampled_map.get(tid, [])
        used = set()
        for txt in reps_texts:
            for idx in topic_indices:
                if idx not in used and docs[idx] == txt:
                    reps_idx.append(idx)
                    used.add(idx)
                    break
        if reps_idx and probs is not None:
            reps_idx.sort(key=lambda i: float(probs[i, tid]), reverse=True)

    if len(reps_idx) < n_docs:
        st.info(
            f"Only {len(reps_idx)} representative document(s) available "
            f"for Topic {tid} (requested {n_docs})."
        )

    if not reps_idx:
        st.write("No documents found for this topic.")
    else:
        for idx in reps_idx:
            chunk_id = metadata["docid"][idx]
            author = metadata["author"][idx]
            work = (
                metadata["title"][idx]
                if "title" in metadata.columns
                else "(unknown work)"
            )
            score = float(probs[idx, tid]) if probs is not None else None
            score_str = f" — Score: {score:.3f}" if score is not None else ""
            st.markdown(f"**Chunk {chunk_id} — {author} — {work}{score_str}**")
            st.write(docs[idx])
            st.write("---")

elif selected_tab == "Label Management":
    _coerce_visible("selected_topic_label")
    st.selectbox(
        "Select topic to analyze",
        visible_choices,
        key="selected_topic_label",
        on_change=_sync_selected_topic,
        args=("selected_topic_label",),
    )
    tid = int(st.session_state["selected_topic_label"].split(":", 1)[0])
    try:
        index = topic_idx_map[tid]
    except KeyError:
        base_info_df = topic_model.get_topic_info().reset_index(drop=True)
        topic_info_df = topic_info_with_custom_names(
            base_info_df, getattr(topic_model, "custom_labels_", None)
        )
        topic_idx_map = {
            int(r["Topic"]): i
            for i, r in enumerate(topic_info_df.to_dict(orient="records"))
        }
        index = topic_idx_map.get(tid, 0)
    st.subheader("Edit custom label")
    if not getattr(topic_model, "custom_labels_", None):
        topic_model.custom_labels_ = [
            f"{int(r['Topic'])}: {r.get('Name', '')}"
            for r in topic_info_df.to_dict(orient="records")
        ]
        topic_model.set_topic_labels(topic_model.custom_labels_)

    current = strip_topic_prefix([topic_model.custom_labels_[index]], [tid])[0]
    st.session_state.setdefault(f"lbl_input_{tid}", current)
    new_lbl = st.text_input("Edit label", key=f"lbl_input_{tid}")
    llm_choices = LLM_REPRESENTATION_OPTIONS
    llm_options = ["None"] + llm_choices

    def _suggest_label(key: str, topic_id: int):
        # always fetch the latest dropdown choice
        llm_key = f"llm_sugg_{topic_id}"
        sel = st.session_state.get(llm_key, "None")
        aspect = None if sel == "None" else sel
        prompt_override = st.session_state.get(
            "sugg_prompt_override", st.session_state.prompt
        )

        # LLM-based label suggestion for single topic
        if aspect and aspect in LLM_REPRESENTATION_OPTIONS:
            with st.spinner(f"Generating label suggestion with {aspect}..."):
                nr_samples = st.session_state.get(f"sugg_nr_docs_{topic_id}", 8)
                reps_idx = sample_representative_indices(
                    st.session_state["docs_full"],
                    st.session_state["metadata_full"],
                    topic_model.topics_,
                    probs,
                    topic_id,
                    nr_samples,
                )
                reps = [st.session_state["docs_full"][i] for i in reps_idx]
                topic_keywords = topic_model.get_topic(topic_id)

                cache_key = xxhash.xxh3_64_hexdigest(
                    f"{model_fingerprint(topic_model)}|{aspect}|{prompt_override}|{topic_id}|{len(reps)}".encode()
                )
                try:
                    suggestion = cached_llm_topic_suggestion(
                        cache_key, aspect, prompt_override, topic_id, topic_model, reps
                    ) or "_".join(w for w, _ in topic_keywords[:3])
                except RetryError:
                    st.error(
                        "The selected LLM provider hit its rate limit. "
                        "Please pick a different model or try again shortly."
                    )
                    return

        # else if it's a non-LLM aspect that actually exists
        elif aspect and aspect in getattr(topic_model, "topic_aspects_", {}):
            rep = topic_model.representation_model.get(aspect)
            if rep and hasattr(rep, "prompt"):
                orig = rep.prompt
                rep.prompt = prompt_override or orig

            nr = st.session_state.nr_label_words
            try:
                labels_all = topic_model.generate_topic_labels(
                    nr_words=nr,
                    topic_prefix=False,
                    separator="_",
                    aspect=aspect,
                )
                # Map to the requested topic_id
                info = topic_model.get_topic_info().reset_index(drop=True)
                tids = info["Topic"].tolist()
                try:
                    idx = tids.index(topic_id)
                    lbl = labels_all[idx]
                except ValueError:
                    lbl = "_".join(w for w, _ in topic_model.get_topic(topic_id)[:3])
                suggestion = lbl
            except (KeyError, IndexError):
                suggestion = "_".join(w for w, _ in topic_model.get_topic(topic_id)[:3])

            if rep and hasattr(rep, "prompt"):
                rep.prompt = orig

        # otherwise default to top 3 words
        else:
            suggestion = "_".join(w for w, _ in topic_model.get_topic(topic_id)[:3])

        st.session_state[key] = suggestion
        # No need to set tab index or rerun - suggestion appears immediately in the text input

    with st.form(key=f"label_suggestion_form_{tid}"):
        st.selectbox(
            "Suggestion model",
            options=llm_options,
            index=min(1, len(llm_options) - 1),
            key=f"llm_sugg_{tid}",
            help="If you select a model, the Prompt below will be used. [DOCUMENTS] and [KEYWORDS] are placeholders.",
        )

        st.number_input(
            "Number of representative docs for suggestion",
            min_value=1,
            max_value=20,
            value=8,
            key=f"sugg_nr_docs_{tid}",
        )

        default_prompt = get_default_prompt(st.session_state["language"])
        initial_prompt = (
            st.session_state.get("sugg_prompt_override")
            or (
                st.session_state.get("applied_prompt") or st.session_state.get("prompt")
            )
            or default_prompt
        )
        with st.expander("Suggestion Prompt", expanded=False):
            st.text_area(
                "Suggestion Prompt",
                value=initial_prompt,
                key="sugg_prompt_override",
                label_visibility="collapsed",
            )

        # Only suggest button inside the form
        st.form_submit_button(
            "Suggest Label",
            on_click=_suggest_label,
            args=(f"lbl_input_{tid}", tid),
        )

    # Outside form: update button runs immediately
    if st.button("Update Label", key=f"update_label_{tid}"):
        new_lbl = st.session_state.get(f"lbl_input_{tid}", "")
        updated_full = f"{tid}: {new_lbl}"
        topic_model.custom_labels_[index] = updated_full
        topic_model.set_topic_labels(topic_model.custom_labels_)
        st.session_state["labels_version"] = (
            st.session_state.get("labels_version", 0) + 1
        )

        # --- Rebuild dropdown option list with updated labels ---
        base_info_df = topic_model.get_topic_info().reset_index(drop=True)
        topic_info_df = topic_info_with_custom_names(
            base_info_df, getattr(topic_model, "custom_labels_", None)
        )
        records = topic_info_df.to_dict(orient="records")
        tids = [int(r["Topic"]) for r in records]
        names = [r.get("CustomName") or r.get("Name") or "" for r in records]
        clean_names = strip_topic_prefix(names, tids)
        topic_choices[:] = [f"{tid0}: {clean_names[i]}" for i, tid0 in enumerate(tids)]
        visible_choices[:] = filter_topic_choices(
            topic_choices, bool(st.session_state.get("show_outliers_option", False))
        )

        st.success(
            "✅ Label updated! Visualizations and selectors have been refreshed."
        )
        st.rerun()  # force reload so selectbox options update this run

    # Show recent update confirmation
    if st.session_state.get(f"label_just_updated_{tid}", False):
        st.info("✅ Label was recently updated")
        # Clear the flag after showing it
        st.session_state[f"label_just_updated_{tid}"] = False

    st.markdown("---")
    col_dl, col_ul, col_rst = st.columns(3)

    with col_dl:
        # Always prepare the CSV data for download
        topic_info = topic_model.get_topic_info().reset_index(drop=True)
        actual_topic_ids = topic_info["Topic"].tolist()

        # Extract clean labels (remove topic_id prefix)
        clean_labels = strip_topic_prefix(topic_model.custom_labels_, actual_topic_ids)

        df_lbl = pl.DataFrame(
            {
                "topic_id": actual_topic_ids,
                "custom_label": clean_labels,
            }
        )

        st.download_button(
            "Download labels CSV",
            df_lbl.write_csv().encode("utf-8"),
            file_name="custom_topic_labels.csv",
            mime="text/csv",
            help="Download current topic labels as CSV for editing and re-upload",
        )

    with col_ul:
        up = st.file_uploader("Upload labels CSV", type="csv")
        if up:
            # Check if we've already processed this file to avoid update loops
            upload_key = f"processed_upload_{hash(up.name + str(up.size))}"
            if upload_key not in st.session_state:
                df_up = pl.read_csv(up)
                if {"topic_id", "custom_label"}.issubset(df_up.columns):
                    # Create mapping from actual topic IDs to custom_labels_ indices
                    topic_info = topic_model.get_topic_info().reset_index(drop=True)
                    actual_topic_ids = topic_info["Topic"].tolist()
                    topic_id_to_index = {
                        tid: i for i, tid in enumerate(actual_topic_ids)
                    }

                    for row in df_up.to_dicts():
                        topic_id = int(row["topic_id"])
                        clean_label = row["custom_label"]

                        # Find the index in custom_labels_ for this topic_id
                        if topic_id in topic_id_to_index:
                            index = topic_id_to_index[topic_id]
                            # Reconstruct full label with topic_id prefix for internal storage
                            full_label = f"{topic_id}: {clean_label}"
                            topic_model.custom_labels_[index] = full_label
                    topic_model.set_topic_labels(topic_model.custom_labels_)
                    st.session_state[upload_key] = True
                    st.session_state.active_tab_index = (
                        2  # Stay on Label Management tab
                    )
                    st.success(
                        "Uploaded & applied labels. Refreshing visualizations..."
                    )
                    st.rerun()
                else:
                    st.error("CSV must contain 'topic_id' and 'custom_label'")

    with col_rst:
        if st.button("Reset to defaults"):
            try:
                apply_first_aspect_labels(topic_model, docs)
                st.session_state["labels_just_reset"] = True
                st.success(
                    "Reset to default labels! Changes will appear in visualizations on next interaction."
                )
            except Exception:
                st.warning("Failed to regenerate labels; keeping previous labels.")

        if st.session_state.get("labels_just_reset", False):
            st.info("✅ Labels were recently reset to defaults")
            st.session_state["labels_just_reset"] = False

"## Topic information"

ttab1, ttab2, ttab3, ttab4, ttab5 = st.tabs(
    [
        "Hierarchical cluster analysis",
        "Similarity matrix heatmap",
        "Intertopic distance map",
        "Per-genre topic distribution",
        "DataMapPlot",
    ]
)


@st.cache_data(show_spinner=True, show_time=True)
def visualize_datamap(
    _unique_id: str,
    docs: list[str],
    embeddings,
    reduced_embeddings,
    hover_text: list[str],
) -> str:
    """
    Generate interactive DataMap HTML with error handling for empty sectors and minimum doc count.
    """
    lang = st.session_state["language"]
    embed = st.session_state["embedding_model"]
    title = f"{lang} Document DataMap ({embed})"
    subtitle = (
        f"A 2D preview of {len(docs)} {lang.lower()} documents clustered by topic"
    )

    # Add validation before attempting to plot
    if len(docs) < 10:
        return "<div>DataMap requires at least 10 documents</div>"

    try:
        fig = topic_model.visualize_document_datamap(
            docs=hover_text,  # This is passed on as-is to the underlying datamapplot and used as hover text
            embeddings=embeddings,
            reduced_embeddings=reduced_embeddings,
            custom_labels=True,
            title=title,
            sub_title=subtitle,
            interactive=True,
            enable_search=True,
            topic_prefix=True,
        )
        return fig._repr_html_()
    except ValueError as e:
        if "array of sample points is empty" in str(e):
            logging.error(f"DataMap plot failed: {e}")
            return "<div>DataMap visualization unavailable: insufficient data distribution</div>"
        else:
            raise


with ttab1:
    st.plotly_chart(
        topic_model.visualize_hierarchy(custom_labels=True),
        key="hierarchy_plot",
    )
    st.markdown(
        """
        **Hierarchical Clustering Interpretation:**
        
        This dendrogram shows how topics are related based on their c-TF-IDF representations:
        - **Vertical axis**: Distance/dissimilarity between topics (lower = more similar)
        - **Horizontal axis**: Individual topics at the leaves
        - **Branches**: Show merge order - topics connected lower are more similar
        - **Applications**: Identify topic groups, find appropriate number of topics, discover topic taxonomies
        """
    )

with ttab2:
    st.plotly_chart(
        topic_model.visualize_heatmap(custom_labels=True),
        key="heatmap_plot",
    )
    st.markdown(
        """
        **Topic Similarity Matrix:**
        
        This heatmap shows pairwise cosine similarities between all topics:
        - **Color intensity**: Darker = more similar (scale 0-1; here implemented as min-max seen values)
        - **Diagonal**: Always 1.0 (topic similarity with itself)
        - **Patterns**: Block structures indicate topic clusters
        - **Use cases**: Identify redundant topics, find topic relationships, validate clustering quality
        """
    )


@st.cache_data(show_spinner=True, show_time=True)
def cached_visualize_topics(_unique_id):
    return topic_model.visualize_topics(custom_labels=True)


@st.cache_data(show_spinner=True, show_time=True)
def cached_visualize_topics_per_class_from_precomputed(
    _unique_id: str, topics_per_class_data: dict, top_n: int
) -> Figure:
    """
    Create visualization from pre-computed topics per class data.

    Args:
        unique_id: Cache key for invalidation.
        topics_per_class_data: Pre-computed topics per class dictionary.
        top_n: Number of top topics to display.

    Returns:
        A Plotly figure from visualize_topics_per_class.
    """
    return topic_model.visualize_topics_per_class(
        topics_per_class_data,
        top_n_topics=top_n,
        custom_labels=True,
    )


with ttab3:
    st.write(cached_visualize_topics(st.session_state.unique_id))
    st.markdown(
        """
        **Intertopic Distance Map:**
        
        2D visualization of topic relationships using multidimensional scaling:
        - **Circle size**: Represents topic prevalence (number of documents)
        - **Distance**: Approximate semantic distance between topics
        - **Overlaps**: Indicate topics with shared vocabulary/concepts
        - **Interpretation**: Similar to PCA/MDS of topic vectors, preserves relative distances
        """
    )

with ttab4:
    if metadata.is_empty() or len(docs) == 0:
        st.info("No metadata available for genre analysis.")
    else:
        doc_count = len(docs)
        if doc_count >= 5000:
            st.warning(
                f"**Genre-topic distribution analysis disabled for large corpora**\n\n"
                f"Current corpus size: {doc_count:,} documents\n\n"
                f"This analysis requires computing topic distributions across all genre-document "
                f"combinations, which becomes computationally expensive with large corpora. "
                f"The analysis is disabled when document count ≥ 5,000 to maintain UI responsiveness.\n\n"
                f"**Alternative approaches:**\n"
                f"- Use document sampling in corpus settings (reduce 'Chunks per document')\n"
                f"- Filter to specific authors/works of interest\n"
                f"- Use the 2D document plot colored by genre for similar insights"
            )
        else:
            # Use pre-computed data if available, otherwise compute on demand
            topics_per_class_data = getattr(
                topic_model, "topics_per_class_genre_", None
            )

            if topics_per_class_data is not None:
                # Use pre-computed data
                st.plotly_chart(
                    cached_visualize_topics_per_class_from_precomputed(
                        st.session_state.unique_id,
                        topics_per_class_data,
                        len(topic_model.topic_sizes_),
                    )
                )
            else:
                # Fallback to on-demand computation with warning
                st.warning(
                    "Genre analysis not pre-computed. Computing now (this may take a moment)..."
                )
                with st.spinner("Computing genre-topic distributions..."):
                    topics_per_class_data = topic_model.topics_per_class(
                        st.session_state["docs_full"],
                        st.session_state["metadata_full"].get_column("genre").to_list(),
                    )
                    fig = topic_model.visualize_topics_per_class(
                        topics_per_class_data,
                        top_n_topics=len(topic_model.topic_sizes_),
                        custom_labels=True,
                    )
                    st.plotly_chart(fig)

            st.markdown(
                """
                **Genre-Topic Distribution Analysis:**
                
                This visualization shows how topics are distributed across different genres:
                - **Bar height**: Proportion of genre documents assigned to each topic
                - **Patterns**: Reveals genre-specific themes and cross-genre topics
                - **Statistical note**: Based on hard topic assignments, not probabilistic distributions
                - **Applications**: Literary analysis, genre characterization, corpus balance assessment

                Note that the keywords for the same topic **differ** between the classes (genres).
                While describing the same topic, they use different vocabulary to express it.
                """
            )

with ttab5:
    authors_list = metadata.get_column("author").to_list()
    titles_list = metadata.get_column("title").to_list()
    filenames_list = metadata.get_column("filename").to_list()

    total_chunks = collections.Counter(filenames_list)

    # Build hover_text with "chunk current/total"
    seen: collections.defaultdict[str, int] = collections.defaultdict(int)
    hover_text: list[str] = []
    for idx, text in enumerate(docs):
        fname = filenames_list[idx]
        seen[fname] += 1
        curr = seen[fname]
        total = total_chunks[fname]
        hover_text.append(
            f"{authors_list[idx]} | {titles_list[idx]} | chunk {curr}/{total}\n{text}"
        )

    # Generate the HTML with custom hover_text
    html_str = visualize_datamap(
        st.session_state.unique_id,
        docs,
        embeddings,
        reduced_embeddings,
        hover_text,
    )

    st.markdown("### Download the interactive DataMap as an HTML file")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="Download DataMap HTML",
            data=html_str,
            file_name=f"document_datamap_{language}_{model_creation_time}.html",
            mime="text/html",
        )

st.markdown("# Topic query")

topic_query = st.text_input("Find topics", key="topic_query")

if topic_query:
    try:
        qtopics, qsim = find_topics_cached(
            st.session_state.unique_id, topic_model, topic_query
        )
        if not qtopics or not qsim:
            qtopics, qsim = [0], [0.0]
    except Exception:
        qtopics, qsim = [0], [0.0]

    # Build df always, even if dummy
    labels: list[str] = []
    for topic_id in qtopics:
        try:
            info_df = topic_model.get_topic_info(topic_id)
            label = info_df.loc[info_df["Topic"] == topic_id, "CustomName"].iloc[0]  # type: ignore[operator]
        except Exception:
            label = "N/A"
        labels.append(label)
    result_df = pl.DataFrame(
        {
            "Topic": qtopics,
            "Label": labels,
            "Similarity": qsim,
        }
    )
    st.dataframe(result_df)
    with st.expander("Explanation"):
        st.markdown(
            """
            ### Topic Query
            
            This search functionality finds topics most semantically similar to your query text.
            
            **How it works:**
            1. Query is embedded using the same sentence transformer as documents
            2. Cosine similarity is calculated between query embedding and topic embeddings
            3. Topics are ranked by similarity score (0-1 scale)
            
            **Similarity scores:**
            - **0.8-1.0**: Very high similarity - query closely matches topic content
            - **0.6-0.8**: Good similarity - query relates to topic themes
            - **0.4-0.6**: Moderate similarity - some conceptual overlap
            - **<0.4**: Low similarity - minimal relationship
            
            **Use cases:**
            - Find topics related to specific concepts or themes
            - Validate topic coherence by querying with expected terms
            - Discover unexpected topic associations
            - Bridge between human interpretation and model representation
            
            **Tips:**
            - Use multiple related terms for more robust results
            - Try both specific and general queries
            - Low scores for obvious queries may indicate preprocessing issues
            """
        )

"## Topic inference"


def visualize_text_cached(doc: str, use_embedding_model: bool):
    """Wrapper for token-level topic visualization…"""
    _topic_distr, topic_token_distr = topic_model.approximate_distribution(
        doc,
        use_embedding_model=use_embedding_model,
        calculate_tokens=True,
        min_similarity=0.001,
        separator="",  # No whitespace for Japanese
    )

    return topic_model.visualize_approximate_distribution(doc, topic_token_distr[0])


use_embedding_model_option = st.checkbox("Use embedding model", False)

example_text = st.text_area(
    f"Token topic approximation using {'embedding' if use_embedding_model_option else 'c-tf-idf'} model. Sample text from {'https://www.aozora.gr.jp/cards/002231/files/62105_76819.html' if language == 'Japanese' else 'https://mysterytribune.com/suspense-novel-excerpt-the-echo-killing-by-christi-daugherty/'}",
    value=(
        """金の羽根
昔あるところに、月にもお日さまにも増して美しい一人娘をお持ちの王さまとお妃さまがおりました。娘はたいそうおてんばで、宮殿中の物をひっくり返しては大騒ぎをしていました。"""
        if st.session_state["language"] == "Japanese"
        else """It was one of those nights.
Early on there was a flicker of hope— a couple of stabbings, a car wreck with potential. But the wounds weren’t serious and the accident was routine. After that it fell quiet."""
    ),
)


st.write(visualize_text_cached(example_text, use_embedding_model_option))
with st.expander("Explanation"):
    st.markdown(
        """
        ### Topic Inference
        
        This visualization shows **token-level topic assignment** for new text, revealing how topics are distributed across individual words/tokens.
        
        **Visualization components:**
        - **Colored tokens**: Each token is colored by its most probable topic
        - **Color intensity**: Represents assignment confidence (darker = more confident)
        - **Topic legend**: Maps colors to topic labels
        
        **Two inference modes:**
        
        **1. c-TF-IDF based (default):**
        - Uses term frequency patterns learned during training
        - Fast but **limited to vocabulary seen during training**
        - Good for texts similar to training corpus
        
        **2. Embedding model based:**
        - Uses semantic similarity via sentence transformer
        - Handles out-of-vocabulary terms better
        - More computationally intensive but more flexible
        - Better for texts with different vocabulary but similar concepts
        
        **Analysis applications:**
        - Understand which words/phrases drive topic assignment
        - Identify topic transitions within text
        - Validate model's topic coherence at token level
        - Debug unexpected document-level classifications
        
        **Interpretation notes:**
        - Frequent topic switches may indicate overlapping topics
        - Consistent coloring suggests strong topical coherence
        - Gray/uncolored tokens are not strongly associated with any topic
        """
    )
