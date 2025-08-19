import logging
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import polars as pl
import xxhash

from topic_modeling_streamlit.nlp_utils import LanguageProcessor, get_default_prompt

logger = logging.getLogger("topic_modeling_streamlit.utils")


@contextmanager
def log_time(
    message: str,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    extra: dict | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that logs start and elapsed time for a code block.

    Args:
        message: short message describing the block (e.g. "embedding.encode")
        logger: optional Logger instance; defaults to module logger
        level: logging level, default INFO
        extra: optional dict of additional fields to include in the log message
    """
    log = (
        logger if logger is not None else logging.getLogger("topic_modeling_streamlit")
    )
    extra = extra or {}
    start = time.perf_counter()
    try:
        log.log(level, "START %s %s", message, extra)
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.log(level, "DONE %s elapsed=%.3fs %s", message, elapsed, extra)


def log_duration(
    name: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to log execution time of a function.

    Usage:
        @log_duration("calculate_model")
        def calculate_model(...):
            ...
    """

    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            start = time.perf_counter()
            log.log(level, "START %s", func_name)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log.log(level, "DONE %s elapsed=%.3fs", func_name, elapsed)

        return wrapper

    return deco


def ensure_tokenizer_types(session_state: dict) -> None:
    """
    Ensure session_state has tokenizer_type and dictionary_type
    parsed from tokenizer_dictionary_option if present.
    """
    tok_dic = session_state.get("tokenizer_dictionary_option")
    if isinstance(tok_dic, str) and "/" in tok_dic:
        tok, dic = tok_dic.split("/", 1)
        session_state["tokenizer_type"] = tok
        session_state["dictionary_type"] = dic
    else:
        # fallback to existing or blanks
        session_state["tokenizer_type"] = session_state.get("tokenizer_type", "")
        session_state["dictionary_type"] = session_state.get("dictionary_type", "")


def propagate_option_keys(session_state: dict) -> None:
    """
    Copy all *_option keys in session_state to bare keys.
    """
    for key in list(session_state.keys()):
        if isinstance(key, str) and key.endswith("_option"):
            bare = key.removesuffix("_option")
            session_state[bare] = session_state[key]


def sync_ui_to_session(session_state: dict) -> None:
    """
    Propagate *_option keys, ensure tokenizer types, and sync surface_filter.
    """
    propagate_option_keys(session_state)
    ensure_tokenizer_types(session_state)
    pattern = session_state.get("surface_filter_option", "").strip()
    session_state.setdefault("surface_filter", pattern or None)


def compute_unique_id(session_state: dict) -> str:
    """
    Compute a unique hash for the session state, skipping 'unique_id' and temp_ keys.
    """

    unique_hash_string = "".join(
        f"{k}{v}"
        for k, v in sorted(session_state.items())
        if k != "unique_id" and not str(k).startswith("temp_")
    ) + str(session_state.get("language", ""))
    return xxhash.xxh3_64_hexdigest(unique_hash_string)


def build_device_choices() -> list[str]:
    """
    Build list of available device strings: CUDA (all GPUs), CUDA-bnb,
    then MPS if available, then CPU.
    """
    import torch

    devices: list[str] = []
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        devices.extend([f"cuda:{i}-bnb" for i in range(torch.cuda.device_count())])
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def apply_ui_options(session_state: dict, reload: bool = False) -> None:
    """
    Synchronize UI *_option keys into bare session_state keys, update device,
    ensure tokenizer types, surface_filter, and applied representation/prompt.

    Args:
        session_state: The st.session_state dict-like object.
        reload: If True, also updates applied_* values and sets computed/reload flags.
    """
    # Track prior language to clear suggestion prompt override when changing
    prev_lang = session_state.get("_prev_language", None)
    curr_lang = session_state.get("language_option", None)
    if prev_lang is not None and curr_lang != prev_lang:
        session_state.pop("sugg_prompt_override", None)
    session_state["_prev_language"] = curr_lang

    # seed bare key defaults from *_option siblings if not present
    session_state.setdefault("chunksize", session_state.get("chunksize_option", 100))
    session_state.setdefault(
        "min_chunksize", session_state.get("min_chunksize_option", 20)
    )
    session_state.setdefault("chunks", session_state.get("chunks_option", 50))

    # if switched to English, force spaCy tokenizer
    if session_state.get("language_option") == "English":
        default_eng = "spaCy/en_core_web_sm"
        valid_eng = {"spaCy/en_core_web_sm", "spaCy/en_core_web_trf"}
        if session_state.get("tokenizer_dictionary_option") not in valid_eng:
            session_state["tokenizer_dictionary_option"] = default_eng

    # propagate *_option → bare keys
    propagate_option_keys(session_state)

    # Always sync device_option down to device
    session_state["device"] = session_state.get(
        "device_option", session_state.get("device", "cpu")
    )

    # Ensure tokenizer types present
    ensure_tokenizer_types(session_state)

    # Compile-ready regex string for filtering tokens
    pattern = session_state.get("surface_filter_option", "").strip()
    session_state["surface_filter"] = pattern or None

    if reload:
        session_state["applied_representation_model"] = session_state.get(
            "representation_model_option",
            session_state.get("applied_representation_model", []),
        )
        session_state["applied_prompt"] = session_state.get(
            "prompt_option", session_state.get("applied_prompt")
        )
        session_state["reload"] = True
        session_state["computed"] = True

    # Set live values from applied_* (used everywhere else)
    session_state["representation_model"] = session_state.get(
        "applied_representation_model", []
    )
    session_state["prompt"] = session_state.get(
        "applied_prompt", session_state.get("prompt")
    )

    # seed the prompt_option from applied prompt (or default)
    lang = session_state.get("language_option", "Japanese")

    session_state.setdefault(
        "prompt_option", session_state.get("applied_prompt", get_default_prompt(lang))
    )
    session_state["prompt"] = session_state.get(
        "applied_prompt", session_state["prompt_option"]
    )

    # unique_id update
    session_state["unique_id"] = compute_unique_id(session_state)


def strip_topic_prefix(labels: list[str], topic_ids: list[int]) -> list[str]:
    """
    Remove the "{topic_id}: " prefix from each label if present.

    Args:
        labels: List of full labels possibly with "{topic_id}: " prefix.
        topic_ids: Corresponding topic IDs for each label.

    Returns:
        List of cleaned label strings without the topic ID prefix.
    """
    clean_labels: list[str] = []
    for tid, full_label in zip(topic_ids, labels):
        expected_prefix = f"{tid}: "
        if isinstance(full_label, str) and full_label.startswith(expected_prefix):
            clean_labels.append(full_label[len(expected_prefix) :])
        else:
            clean_labels.append(full_label)
    return clean_labels


def validate_vectorizer(vectorizer, language: str):
    """
    Validate that the vectorizer has a callable tokenizer and yields tokens for sample docs.
    Falls back to a known-good LanguageProcessor if invalid/empty.
    """

    if not hasattr(vectorizer, "tokenizer") or not callable(vectorizer.tokenizer):
        raise ValueError(
            f"Vectorizer has no callable tokenizer — check tokenizer settings for {language}"
        )

    # Probe tokenization
    sample_text = "テスト文章" if language == "Japanese" else "This is a test document."
    try:
        sample_tokens = vectorizer.tokenizer(sample_text)
    except Exception as e:
        raise ValueError(f"Tokenizer failed to run: {e}")

    if not sample_tokens:
        logging.warning(
            f"Tokenizer '{getattr(vectorizer, 'tokenizer', None)}' returned no tokens; using fallback defaults."
        )
        if language == "English":
            vectorizer = LanguageProcessor(
                language="English",
                tokenizer_type="spaCy",
                dictionary_type="en_core_web_sm",
            ).vectorizer
        else:
            vectorizer = LanguageProcessor(
                language="Japanese",
                tokenizer_type="MeCab",
                dictionary_type="近現代口語小説UniDic",
            ).vectorizer
    return vectorizer


def get_metadata_df(language: str, ui_warn: bool = False):
    """
    Unified metadata loading for Japanese or English corpora.

    Args:
        language: "Japanese" or "English"
        ui_warn: if True, emit st.error on missing files; else just return None
    """

    if language == "Japanese":
        groups = Path("Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv")
        if not groups.exists():
            if ui_warn:
                import streamlit as st

                st.error(
                    "Missing Aozora metadata: Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv\n"
                    "Please download into Aozora-Bunko-Fiction-Selection-2022-05-30/ with groups.csv and Plain/*.txt."
                )
            return None
        df = pl.read_csv(str(groups), separator="\t")
        if ui_warn:
            # Drop original 'author'/'title' if we are replacing them to avoid duplicates
            to_rename = {}
            if "author_ja" in df.columns:
                if "author" in df.columns:
                    df = df.drop("author")
                to_rename["author_ja"] = "author"
            if "title_ja" in df.columns:
                if "title" in df.columns:
                    df = df.drop("title")
                to_rename["title_ja"] = "title"
            if to_rename:
                df = df.rename(to_rename)
        return df
    else:
        se_meta = Path("standard_ebooks_metadata.csv")
        if not se_meta.exists():
            if ui_warn:
                import streamlit as st

                st.error(
                    "Missing Standard Ebooks metadata: standard_ebooks_metadata.csv\n"
                    "Please place CSV and texts under standard-ebooks-selection/*.txt."
                )
            return None
        return pl.read_csv(str(se_meta))


def build_representation_model(
    model_name: str,
    language: str,
    prompt: str | None,
    device: str,
    model_kwargs: dict | None = None,
):
    """
    Factory for creating a BERTopic representation model instance from its string name.
    """
    # Inline imports to prevent circular import error
    import spacy
    import tiktoken
    from bertopic.representation import (
        KeyBERTInspired,
        LiteLLM,
        MaximalMarginalRelevance,
        PartOfSpeech,
    )

    from topic_modeling_streamlit.nlp_utils import CleanTextGeneration
    from topic_modeling_streamlit.transformers_utils import load_transformers_model

    if model_name == "KeyBERTInspired":
        return KeyBERTInspired()
    if model_name == "MaximalMarginalRelevance":
        return MaximalMarginalRelevance(diversity=0.5)
    if model_name == "PartOfSpeech":
        spacy_model_name = (
            "en_core_web_sm" if language == "English" else "ja_core_news_sm"
        )
        return PartOfSpeech(
            spacy.load(spacy_model_name, disable=["parser", "ner", "lemmatizer"])
        )
    if model_name.startswith("local:"):
        actual_model = model_name.removeprefix("local:")
        generator, tokenizer = load_transformers_model(
            model_name=actual_model,
            device=device,
            max_new_tokens=20,
            model_kwargs=model_kwargs,
        )
        system_msg = f"""You are a topic analyzer that condenses keywords and representative documents into precise labels...
Topic labels must be in {language} and no longer than around 5 words./no_think"""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt or ""},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return CleanTextGeneration(
            model=generator,
            prompt=full_prompt,
            nr_docs=8,
            doc_length=100,
            tokenizer="vectorizer",
            pipeline_kwargs={},
        )
    if model_name.startswith("litellm:"):
        actual_model = model_name.removeprefix("litellm:")
        return LiteLLM(actual_model)
    if model_name == "gpt-4.1-nano":
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        return LiteLLM("azure/gpt-4.1-nano")
    return None


def filter_topic_choices(topic_choices: list[str], show_outliers: bool) -> list[str]:
    """Return list of topic choice strings, optionally filtering out outlier topic -1."""
    return [c for c in topic_choices if show_outliers or not c.startswith("-1:")]


def validate_chunking_settings(chunksize: int, min_chunksize: int):
    """Raise ValueError if chunk size settings are invalid."""
    if min_chunksize >= chunksize:
        raise ValueError(
            f"Minimum chunk size ({min_chunksize}) must be less than max chunk size ({chunksize})."
        )


def sample_representative_indices(
    docs_full, metadata_full, topics, probs, tid, nr_docs
):
    """Return list of doc indices representing given topic id."""
    from topic_modeling_streamlit.nlp_utils import stratified_weighted_document_sampling

    reps_map = stratified_weighted_document_sampling(
        docs_full,
        metadata_full,
        topics,
        probs,
        nr_docs=nr_docs,
        stratify_by="genre",
        seed=42,
    )
    reps_texts = reps_map.get(tid, [])
    topic_indices = [i for i, t in enumerate(topics) if t == tid]
    used, reps_idx = set(), []
    for txt in reps_texts:
        for idx in topic_indices:
            if idx not in used and docs_full[idx] == txt:
                reps_idx.append(idx)
                used.add(idx)
                break
    return reps_idx


def get_umap_model(device: str, n_neighbors: int = 10, n_components: int = 2):
    """Return a configured UMAP model, trying torchdr GPU/MPS first then CPU fallback."""
    try:
        import torch
        from torchdr import UMAP as TorchdrUMAP

        if device.startswith("cuda") and torch.cuda.is_available():
            return TorchdrUMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                device="cuda",
            )
        if (
            device == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return TorchdrUMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                device="mps",
            )
        raise ImportError
    except (ImportError, ModuleNotFoundError, RuntimeError):
        from umap import UMAP as CPUUMAP

        return CPUUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
        )
