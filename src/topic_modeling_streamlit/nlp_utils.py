import logging
import os
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import litellm.exceptions
import numpy as np
import pandas as pd
import polars as pl
import pysbd
import regex as re
import rhoknp
import spacy
import sudachipy
import torch
import xxhash
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    LiteLLM,
    TextGeneration,
)
from filelock import FileLock
from fugashi import Tagger
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential
from transformers.pipelines import Pipeline

from topic_modeling_streamlit.data_lib import (
    create_corpus_core,
    get_metadata,
)
from topic_modeling_streamlit.transformers_utils import load_transformers_model
from topic_modeling_streamlit.utils import (
    build_representation_model,
    get_umap_model,
    log_time,
    validate_vectorizer,
)

_original_litellm_extract = LiteLLM.extract_topics


def llm_extract(
    aspect: str,
    prompt: str,
    model: BERTopic,
    docs_sample: list[str],
    tids: list[int],
    device: str,
) -> dict[int, list[tuple[str, float]]]:
    """
    Core LLM extraction without Streamlit. Builds a minimal DataFrame and calls
    the appropriate representation to extract per-topic labels/scores.
    """
    with log_time(
        "llm_extract_call",
        extra={"aspect": aspect, "n_docs_sample": len(docs_sample), "device": device},
    ):
        rep = get_llm_representation(aspect, prompt, device)

        docs_df = pd.DataFrame(
            {
                "Document": docs_sample[: len(model.topics_)],
                "Topic": model.topics_[: len(docs_sample)],
            }
        )
        # Support both list[int] and dict[int, list[tuple[str, float]]] topics param
        topics_arg: Any
        if isinstance(tids, dict):
            topics_arg = tids
        else:
            topics_arg = tids
        return rep.extract_topics(model, docs_df, model.c_tf_idf_, topics_arg)


def llm_topic_suggestion(
    aspect: str,
    prompt: str,
    topic_id: int,
    model: BERTopic,
    reps_docs: list[str],
    device: str,
) -> str:
    """
    Core single-topic LLM suggestion without Streamlit dependencies.
    """
    rep = get_llm_representation(aspect, prompt, device)

    if not reps_docs:
        kw = model.get_topic(topic_id)[:3]
        return "_".join(w for w, _ in kw)
    docs_df = pd.DataFrame(
        {"Document": reps_docs, "Topic": [topic_id] * len(reps_docs)}
    )
    topics_dict = {topic_id: model.get_topic(topic_id)}
    with log_time(
        "llm_topic_suggestion_call",
        extra={"aspect": aspect, "topic_id": topic_id, "n_docs": len(reps_docs)},
    ):
        out = rep.extract_topics(model, docs_df, model.c_tf_idf_, topics_dict)
    return out.get(topic_id, [("", 0.0)])[0][0]


def get_default_prompt(language: str) -> str:
    """
    Return the default LLM-prompt template for topic suggestions in the given language.
    """
    return f"""I have a topic that contains the following representative documents:

[DOCUMENTS]

The topic is described by the following keywords: [KEYWORDS]

Based the keywords and the documents that show their context, return a short label of the topic in {language} in less than 30 characters and nothing else.
The documents are only a sample for reference and may come from different works.
Greater weight should be put on explaining the keywords with common themes."""


def model_fingerprint(model: BERTopic) -> str:
    """
    Lightweight fingerprint used in cache keys.

    Captures:
    - c-TF-IDF shape
    - vectorizer ngram_range, max_df
    - order-insensitive digest of vocabulary keys and size
    - topics list hash
    - nr_topics
    """
    ctf = getattr(model, "c_tf_idf_", None)
    ctf_shape = getattr(ctf, "shape", (0, 0))

    vec = getattr(model, "vectorizer_model", None)
    try:
        ngram_range = getattr(vec, "ngram_range", None)
    except Exception:
        ngram_range = None
    try:
        max_df = getattr(vec, "max_df", None)
    except Exception:
        max_df = None

    # Build a stable, order-insensitive digest of the vocabulary keys
    try:
        vocab_dict = getattr(vec, "vocabulary_", {}) or {}
        vocab_keys_sorted = sorted(vocab_dict.keys())
        vocab_digest = xxhash.xxh3_64_hexdigest(
            pickle.dumps(vocab_keys_sorted, pickle.HIGHEST_PROTOCOL)
        )
        vocab_size = len(vocab_keys_sorted)
    except Exception:
        vocab_digest = "0"
        vocab_size = 0

    topics_list = getattr(model, "topics_", []) or []
    topics_hash = xxhash.xxh3_64_hexdigest(
        pickle.dumps(topics_list, pickle.HIGHEST_PROTOCOL)
    )

    payload = (
        ctf_shape,
        ngram_range,
        max_df,
        vocab_size,
        vocab_digest,
        topics_hash,
        getattr(model, "nr_topics", None),
    )
    return xxhash.xxh3_64_hexdigest(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))


def get_embedding_model_options(language: str) -> list[str]:
    """
    Get language-appropriate embedding model options.

    Args:
        language: "Japanese" or "English".

    Returns:
        List of model names/paths suitable for the language.

    Examples:
        >>> "cl-nagoya/ruri-v3-30m" in get_embedding_model_options("Japanese")
        True
        >>> "prdev/mini-gte" in get_embedding_model_options("English")
        True
    """
    if language == "Japanese":
        return [
            "cl-nagoya/ruri-v3-30m",
            "sbintuitions/sarashina-embedding-v1-1b",
            "StyleDistance/mstyledistance",
            "Qwen/Qwen3-Embedding-0.6B",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-large",
        ]
    return [
        "prdev/mini-gte",
        "Qwen/Qwen3-Embedding-0.6B",
        "AnnaWegmann/Style-Embedding",
        "StyleDistance/mstyledistance",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
        "thenlper/gte-base",
    ]


def topic_info_with_custom_names(
    info_df: pd.DataFrame, custom_labels: list[str] | None
) -> pd.DataFrame:
    """
    Ensure CustomName column reflects provided custom labels.

    Args:
        info_df: BERTopic.get_topic_info().reset_index(drop=True) DataFrame.
        custom_labels: Optional labels list to overwrite CustomName.

    Returns:
        Copy of info_df with CustomName set if labels provided.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"Topic": [0, 1], "Name": ["0: A", "1: B"]})
        >>> out = topic_info_with_custom_names(df, ["0: X", "1: Y"])
        >>> out["CustomName"].tolist()
        ['0: X', '1: Y']
    """
    out = info_df.reset_index(drop=True).copy()
    if custom_labels:
        out["CustomName"] = custom_labels
    return out


def generate_labels_core(
    model: BERTopic,
    nr_words: int,
    aspect: str | None,
    prompt: str | None,
    device: str | None = None,
    docs_sample: list[str] | None = None,
) -> list[str]:
    """
    Pure core for generating labels. For LLM aspects, use llm_extract;
    otherwise delegate to BERTopic.generate_topic_labels.
    """
    try:
        with log_time("generate_topic_labels_bertopic", extra={"nr_words": nr_words}):
            raw_labels = model.generate_topic_labels(
                nr_words=nr_words,
                topic_prefix=False,
                separator="_",
                aspect=aspect,
            )
    except KeyError:
        if aspect and (aspect.startswith("local:") or aspect.startswith("litellm:")):
            info = model.get_topic_info().reset_index(drop=True)
            tids = info["Topic"].tolist()
            sample = list(docs_sample or [])
            if not sample:
                try:
                    reps = model.get_representative_docs()
                    pool: list[str] = []
                    for tid in tids:
                        pool.extend(reps.get(tid, [])[:1])
                    sample = pool or getattr(model, "docs_", []) or []
                except Exception:
                    sample = getattr(model, "docs_", []) or []
            with log_time("llm_extract_labels", extra={"aspect": aspect}):
                try:
                    rep = get_llm_representation(aspect, prompt or "", device or "cpu")
                    docs_df = pd.DataFrame(
                        {
                            "Document": sample[: len(model.topics_)],
                            "Topic": model.topics_[: len(sample)],
                        }
                    )
                    topics_arg = tids
                    out = _retry_extract_topics_safe(
                        rep, model, docs_df, model.c_tf_idf_, topics_arg
                    )
                except RateLimitError as e:
                    logging.warning(
                        f"[generate_labels_core] LLM aspect {aspect} rate-limited: {e}"
                    )
                    out = {
                        tid: [("_".join(w for w, _ in model.get_topic(tid)[:3]), 1.0)]
                        for tid in tids
                    }
                except ServiceUnavailableError as e:
                    logging.warning(
                        f"[generate_labels_core] LLM aspect {aspect} service unavailable: {e}"
                    )
                    out = {
                        tid: [("_".join(w for w, _ in model.get_topic(tid)[:3]), 1.0)]
                        for tid in tids
                    }
                except Exception as e:
                    logging.warning(
                        f"[generate_labels_core] LLM aspect {aspect} failed: {e}"
                    )
                    out = {
                        tid: [("_".join(w for w, _ in model.get_topic(tid)[:3]), 1.0)]
                        for tid in tids
                    }
            raw_labels = [out.get(t, [("", 0.0)])[0][0] for t in tids]
        else:
            raise
    info = model.get_topic_info().reset_index(drop=True)
    topic_ids = info["Topic"].tolist()
    # Normalize lengths to avoid index mismatches when certain topics (e.g., -1) have no labels
    if len(raw_labels) != len(topic_ids):
        padded: list[str] = []
        for tid in topic_ids:
            idx = len(padded)
            if idx < len(raw_labels) and raw_labels[idx]:
                padded.append(raw_labels[idx])
            else:
                kw = model.get_topic(tid)[:3]
                padded.append("_".join(w for w, _ in kw) if kw else f"{tid}")
        raw_labels = padded
    return [f"{tid}: {lbl.rstrip('_')}" for tid, lbl in zip(topic_ids, raw_labels)]


def apply_first_aspect_labels_core(
    topic_model: BERTopic,
    aspect: str,
    nr_words: int,
    prompt: str | None,
    device: str | None = None,
) -> list[str]:
    """
    Core function to compute and set custom labels for the first aspect.
    """
    with log_time(
        "apply_first_aspect_labels", extra={"aspect": aspect, "nr_words": nr_words}
    ):
        labels = generate_labels_core(topic_model, nr_words, aspect, prompt, device)
        topic_model.custom_labels_ = labels
        topic_model.set_topic_labels(labels)
        return labels


def ensure_aspect_core(
    topic_model: BERTopic,
    aspect: str,
    prompt: str | None,
    language: str,
    docs: list[str],
    device: str | None = None,
) -> None:
    """
    Compute and inject representation for an aspect without Streamlit dependencies.
    """
    info = topic_model.get_topic_info().reset_index(drop=True)
    all_tids: list[int] = info["Topic"].tolist()
    tids: list[int] = all_tids[:]

    if aspect.startswith("local:") or aspect.startswith("litellm:"):
        # Build a non-empty sample: prefer provided docs; else use representative docs per topic; else model.docs_
        sample = list(docs or [])
        if not sample:
            try:
                reps = topic_model.get_representative_docs()
                pool: list[str] = []
                for tid in tids:
                    pool.extend(reps.get(tid, [])[:1])
                sample = pool or getattr(topic_model, "docs_", []) or []
            except Exception:
                sample = getattr(topic_model, "docs_", []) or []
        with log_time("llm_extract_aspect", extra={"aspect": aspect}):
            try:
                rep = get_llm_representation(aspect, prompt or "", device or "cpu")
                docs_df = pd.DataFrame(
                    {
                        "Document": sample[: len(topic_model.topics_)],
                        "Topic": topic_model.topics_[: len(sample)],
                    }
                )
                topics_dict = tids
                out = _retry_extract_topics_safe(
                    rep, topic_model, docs_df, topic_model.c_tf_idf_, topics_dict
                )
            except RateLimitError as e:
                logging.warning(
                    f"[ensure_aspect_core] Skipping aspect {aspect} due to rate limit: {e}"
                )
                out = {tid: [] for tid in tids}
            except ServiceUnavailableError as e:
                logging.warning(
                    f"[ensure_aspect_core] Skipping aspect {aspect} due to service unavailability: {e}"
                )
                out = {tid: [] for tid in tids}
            except Exception as e:
                logging.warning(
                    f"[ensure_aspect_core] Skipping aspect {aspect} due to error: {e}"
                )
                out = {tid: [] for tid in tids}
        topic_model.topic_aspects_.setdefault(aspect, {})
        for tid in tids:
            topic_model.topic_aspects_[aspect][tid] = out.get(tid, [])
        return

    with log_time("ensure_aspect_nonllm", extra={"aspect": aspect}):
        rep = getattr(topic_model, "representation_model", {}) or {}
        if aspect not in rep:
            rm = build_representation_model(aspect, language, prompt, device)
            if rm:
                rep[aspect] = rm
        topic_model.representation_model = rep

        amap = topic_model.topic_aspects_.setdefault(aspect, {})
        for tid in all_tids:
            if tid not in amap or not amap[tid]:
                top_kw = topic_model.get_topic(tid)
                if top_kw:
                    seed_word = top_kw[0][0]
                    amap[tid] = [(seed_word, 1.0)]
                else:
                    amap[tid] = []

        _ = topic_model.generate_topic_labels(
            nr_words=3,
            topic_prefix=False,
            separator="_",
            aspect=aspect,
        )
        for _tid in tids:
            _ = topic_model.get_topic(_tid)


@retry(
    retry=retry_if_exception_type(litellm.exceptions.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_delay(60),
)
def _retry_litellm_extract_topics(self, topic_model, documents, c_tf_idf, topics):
    return _original_litellm_extract(self, topic_model, documents, c_tf_idf, topics)


LiteLLM.extract_topics = _retry_litellm_extract_topics


@retry(
    retry=retry_if_exception_type(litellm.exceptions.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_delay(60),
)
def _retry_extract_topics_safe(rep, topic_model, documents, c_tf_idf, topics):
    """
    Retry wrapper for any rep.extract_topics(...) that should back off on RateLimitError.
    """
    return rep.extract_topics(topic_model, documents, c_tf_idf, topics)


# If a lock file is older than this (seconds), we consider it stale and delete it
STALE_LOCK_THRESHOLD = 60 * 60  # 1 hour


def _cleanup_stale_lock(lock_path: Path, max_age: int = STALE_LOCK_THRESHOLD) -> None:
    """
    Remove the lock file if its mtime is older than max_age seconds.
    """
    try:
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > max_age:
                logging.warning(f"Removing stale lock {lock_path} (age={age:.0f}s)")
                lock_path.unlink()
    except Exception as e:
        logging.warning(f"Failed cleaning stale lock {lock_path}: {e}")


class CleanTextGeneration(TextGeneration):
    """
    TextGeneration wrapper that removes thinking tags from LLM outputs and logs prompts after creation.

    Some LLMs (especially instruction-tuned models) may include XML-style
    thinking tags like <think>...</think> in their outputs. This wrapper
    automatically strips these tags to provide clean topic labels.

    Inherits all parameters and methods from TextGeneration.

    Examples:
        >>> generator = pipeline('text-generation', model='some-model')
        >>> clean_gen = CleanTextGeneration(generator)
        >>> # Output will have <think> tags removed automatically
    """

    _orig_model: Callable[..., Any]
    model: Callable[..., Any]

    def __init__(
        self,
        model: Union[str, Pipeline],
        prompt: str | None = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        random_state: int = 42,
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: Union[str, Callable] | None = None,
    ):
        super().__init__(
            model=model,
            prompt=prompt,
            pipeline_kwargs=pipeline_kwargs,
            random_state=random_state,
            nr_docs=nr_docs,
            diversity=diversity,
            doc_length=doc_length,
            tokenizer=tokenizer,
        )
        self._orig_model = self.model

        def model_with_logging(prompt_text: str, **kwargs):
            logging.info(f"[LLM Prompt]\n{prompt_text}\n{'-' * 80}")
            return self._orig_model(prompt_text, **kwargs)

        self.model = model_with_logging

    def _create_prompt(self, docs, topic, topics):
        prompt = super()._create_prompt(docs, topic, topics)
        logging.info(f"[Created Prompt for Topic {topic}]\n{prompt}\n{'-' * 80}")
        return prompt

    def extract_topics(
        self,
        topic_model,
        documents,
        c_tf_idf,
        topics,
    ) -> dict[int, list[tuple[str, float]]]:
        # Ensure topics is a dictionary if passed as a list
        if isinstance(topics, list):
            # Create proper topic structure with keywords from topic_model
            topics_dict = {}
            for tid in topics:
                # Get keywords for this topic from the topic model
                topic_keywords = topic_model.get_topic(tid)
                if topic_keywords:
                    topics_dict[tid] = topic_keywords
                else:
                    # Fallback to empty list if no keywords
                    topics_dict[tid] = []
            topics = topics_dict
        # 1) call parent to generate raw topic descriptions (and log prompts)
        raw = super().extract_topics(topic_model, documents, c_tf_idf, topics)
        # 2) strip any <think> tags from the LLM outputs
        cleaned: dict[int, list[tuple[str, float]]] = {}
        for topic_id, descs in raw.items():
            new_descs: list[tuple[str, float]] = []
            for text, score in descs:
                txt = re.sub(r"</?think>", "", text).strip()
                new_descs.append((txt, score))
            cleaned[topic_id] = new_descs
        return cleaned


def get_llm_representation(
    aspect: str, prompt: str, device: str
) -> CleanTextGeneration:
    """
    Return a CleanTextGeneration wrapper for a local HF LLM or LiteLLM wrapper.
    """
    if aspect.startswith("litellm:"):
        actual = aspect.removeprefix("litellm:")
        print("litellm", prompt)
        return LiteLLM(actual, prompt=prompt)

    if aspect.startswith("local:"):
        model_name = aspect.removeprefix("local:")

        generator, tokenizer = load_transformers_model(
            model_name=model_name,
            device=device,
            max_new_tokens=20,
        )

        system_msg = f"""
You are a topic analyzer that condenses keywords and representative documents into precise labels that describe their overall topic.
Only return the topic label itself and nothing else.
Topic labels must be in {prompt} and no longer than around 5 words./no_think
        """.strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
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

    raise ValueError(f"Unsupported LLM aspect: {aspect!r}")


class Dictionary(ABC):
    def __init__(self, name: str):
        self.name = name


class SpacyDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class FugashiDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class SudachipyDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class JumanppDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class Tokenizer(ABC):
    def __init__(
        self,
        dictionary: Dictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        self.dictionary = dictionary
        self.segmenter = segmenter
        self.features = features
        self.pos_filter = pos_filter
        self.whitespace_rx = whitespace_rx

    @abstractmethod
    def get_features(self, t) -> str:
        pass

    @abstractmethod
    def tokenize(self, s: str) -> list[str]:
        pass


class SpacyTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: SpacyDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        self.model = spacy.load(self.dictionary.name)

    def get_features(self, t) -> str:
        mapping = {
            "orth": lambda t: t.orth_,
            "pos1": lambda t: t.pos_,
            "lemma": lambda t: t.lemma_,
            "text_with_ws": lambda t: t.text_with_ws,
        }
        if not self.features:
            return t.text
        else:
            return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        sentences = self.segmenter.segment(s)
        return [
            self.get_features(t)
            for doc in self.model.pipe(sentences, disable=["parser", "ner"])
            for t in doc
            if not self.whitespace_rx.match(t.orth_)
            and not (self.pos_filter and t.pos_ in self.pos_filter)
        ]


class FugashiTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        self.model = Tagger(dictionary.name)

    def get_features(self, t) -> str:
        if not self.features:
            return t.surface

        mapping = {
            "orth": lambda t: t.surface,
            "pos1": lambda t: t.feature.pos1,
            "lemma": lambda t: t.feature.lemma or t.surface,
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence)
            if not self.whitespace_rx.match(t.surface)
            and not (self.pos_filter and t.feature.pos1 in self.pos_filter)
        ]


class SudachipyTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        match dictionary.name[-1]:
            case "A":
                self.mode = sudachipy.SplitMode.A
            case "B":
                self.mode = sudachipy.SplitMode.B
            case "C":
                self.mode = sudachipy.SplitMode.C

        sudachi = sudachipy.Dictionary(dict="full").create()  # TODO
        self.model = sudachi.tokenize

    def get_features(self, t) -> str:
        if not self.features:
            return t.surface()
        mapping = {
            "orth": lambda t: t.surface(),
            "pos1": lambda t: t.part_of_speech()[0],
            "lemma": lambda t: t.normalized_form(),
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence, self.mode)
            if not self.whitespace_rx.match(t.surface())
            and not (self.pos_filter and t.part_of_speech()[0] in self.pos_filter)
        ]


class JumanppTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        self.model = rhoknp.Jumanpp().apply_to_sentence

    def get_features(self, t) -> str:
        if not self.features:
            return t.text

        mapping = {
            "orth": lambda t: t.text,
            "pos1": lambda t: t.pos,
            "lemma": lambda t: t.lemma,
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence).morphemes
            if not self.whitespace_rx.match(t.text)
            and not (self.pos_filter and t.pos in self.pos_filter)
        ]


TOKENIZER_MAP = {
    "spaCy": SpacyTokenizer,
    "MeCab": FugashiTokenizer,
    "Sudachi": SudachipyTokenizer,
    "Juman++": JumanppTokenizer,
}

MECAB_DICDIR_CWJ = os.environ.get("MECAB_DICDIR_CWJ")
MECAB_DICDIR_CSJ = os.environ.get("MECAB_DICDIR_CSJ")
MECAB_DICDIR_NOVEL = os.environ.get("MECAB_DICDIR_NOVEL")

DICTIONARY_MAP = {
    "近現代口語小説UniDic": FugashiDictionary(
        f"-d {MECAB_DICDIR_NOVEL} -r {MECAB_DICDIR_NOVEL}/dicrc"
    ),
    "UniDic-CWJ": FugashiDictionary(
        f"-d {MECAB_DICDIR_CWJ} -r {MECAB_DICDIR_CWJ}/dicrc"
    ),
    "UniDic-CSJ": FugashiDictionary(
        f"-d {MECAB_DICDIR_CSJ} -r {MECAB_DICDIR_CSJ}/dicrc"
    ),
    "SudachiDict-full/A": SudachipyDictionary("SudachiDict-full/A"),
    "SudachiDict-full/B": SudachipyDictionary("SudachiDict-full/B"),
    "SudachiDict-full/C": SudachipyDictionary("SudachiDict-full/C"),
    "ja_core_news_sm": SpacyDictionary("ja_core_news_sm"),
    "Jumandict": JumanppDictionary(""),
    "en_core_web_sm": SpacyDictionary("en_core_web_sm"),
    "en_core_web_trf": SpacyDictionary("en_core_web_trf"),
}


class LanguageProcessor:
    """
    Unified text processing pipeline for Japanese and English documents.

    Handles tokenization, vectorization, and feature extraction using various
    NLP libraries (MeCab, Sudachi, spaCy, Juman++) with consistent interface.
    Supports n-gram extraction, POS filtering, and custom feature combinations.

    Args:
        language: Target language ("Japanese" or "English").
        tokenizer_type: Tokenization engine to use:
            - Japanese: "MeCab", "Sudachi", "spaCy", "Juman++"
            - English: "spaCy"
        dictionary_type: Dictionary/model for the tokenizer:
            - MeCab: "近現代口語小説UniDic", "UniDic-CWJ"
            - Sudachi: "SudachiDict-full/A", "/B", "/C"
            - spaCy: "ja_core_news_sm", "en_core_web_sm", etc.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        max_df: Maximum document frequency for vocabulary (0.0-1.0).
        features: List of token features to extract:
            - "orth": Surface form (original text)
            - "lemma": Dictionary/base form
            - "pos1": Part-of-speech tag
            Multiple features are joined with "/".
        pos_filter: Set of POS tags to exclude from tokenization.
        vocabulary: Pre-defined vocabulary for vectorizer (for loading saved models).

    Attributes:
        tokenizer: Configured tokenizer instance.
        vectorizer: CountVectorizer configured with the tokenizer.

    Examples:
        >>> # Japanese processing with MeCab
        >>> proc = LanguageProcessor(
        ...     language="Japanese",
        ...     tokenizer_type="MeCab",
        ...     dictionary_type="UniDic-CWJ",
        ...     features=["lemma", "pos1"]
        ... )
        >>> tokens = proc.tokenizer.tokenize("これはテストです。")

        >>> # English processing
        >>> eng_proc = LanguageProcessor(
        ...     language="English",
        ...     tokenizer_type="spaCy",
        ...     dictionary_type="en_core_web_sm"
        ... )
    """

    def __init__(
        self,
        language: str = "Japanese",
        tokenizer_type: str = "MeCab",
        dictionary_type: str = "UniDic-CWJ",
        ngram_range: tuple[int, int] = (1, 1),
        max_df: float = 0.8,
        features: list[str] | None = None,
        pos_filter: set[str] | None = None,
        surface_filter: str | re.Pattern | None = None,
        vocabulary: dict[str, int] | None = None,
    ):
        # useprovided regex (string or compiled) to drop matching tokens
        if surface_filter is None:
            self.surface_filter = None
        elif hasattr(surface_filter, "search"):  # already a re.Pattern
            self.surface_filter = surface_filter
        else:
            self.surface_filter = re.compile(surface_filter, re.UNICODE)
        self.language = language
        self.segmenter = pysbd.Segmenter(
            language=self.language[:2].lower(), clean=False
        )
        self.whitespace_rx: re.Pattern = re.compile(r"^\s*$")  # TODO

        self.features = features
        self.pos_filter = pos_filter

        # Pickle helpers:
        self.tokenizer_type = tokenizer_type
        self.dictionary_type = dictionary_type

        self.dictionary = DICTIONARY_MAP[dictionary_type]
        self.tokenizer = TOKENIZER_MAP[tokenizer_type](
            self.dictionary,
            self.segmenter,
            self.features,
            self.pos_filter,
            self.whitespace_rx,
        )
        # wrap the tokenize() method to drop any exact matches
        orig = self.tokenizer.tokenize

        def tokenize_filtered(text: str) -> list[str]:
            toks = orig(text)
            if self.surface_filter:
                # remove any token matching the regex
                toks = [t for t in toks if not self.surface_filter.search(t)]
            return toks

        self.tokenizer.tokenize = tokenize_filtered

        self.ngram_range = ngram_range
        self.max_df = max_df

        self.vocabulary = vocabulary

        if self.vocabulary is not None:
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_df=self.max_df,
                tokenizer=self.tokenizer.tokenize,
                lowercase=False if language == "Japanese" else True,
                token_pattern=r".+",  # i.e. no pattern; take all
                vocabulary=vocabulary,
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_df=self.max_df,
                tokenizer=self.tokenizer.tokenize,
                lowercase=False if language == "Japanese" else True,
                token_pattern=r".+",  # i.e. no pattern; take all
            )

        if self.language == "English":
            assert isinstance(self.tokenizer, SpacyTokenizer)
            assert isinstance(self.dictionary, SpacyDictionary)
        elif self.language == "Japanese":
            assert isinstance(
                self.tokenizer,
                (
                    FugashiTokenizer,
                    SudachipyTokenizer,
                    SpacyTokenizer,
                    JumanppTokenizer,
                ),
            )
            assert isinstance(
                self.dictionary,
                (
                    FugashiDictionary,
                    SudachipyDictionary,
                    SpacyDictionary,
                    JumanppDictionary,
                ),
            )

    def __repr__(self):
        return f"JapaneseTagger<{self.tokenizer_type},{self.dictionary_type},features={self.features},pos_filter={self.pos_filter}>"

    def __getstate__(self):
        logging.error("getstate")
        return (
            self.language,
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.max_df,
            self.features,
            self.pos_filter,
            self.vocabulary,
        )

    def __setstate__(self, state):
        logging.error("setstate")
        (
            self.language,
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.max_df,
            self.features,
            self.pos_filter,
            self.vocabulary,
        ) = state
        # Restore from initial params via class‐bound __init__
        LanguageProcessor.__init__(
            self,
            language=self.language,
            tokenizer_type=self.tokenizer_type,
            dictionary_type=self.dictionary_type,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            features=self.features,
            pos_filter=self.pos_filter,
            vocabulary=self.vocabulary,
        )


def stratified_weighted_document_sampling(
    docs: list[str],
    metadata: pl.DataFrame,
    topics: list[int],
    probs: np.ndarray,
    nr_docs: int = 5,
    stratify_by: str = "genre",
    seed: int | None = None,
) -> dict[int, list[str]]:
    """
    For each topic, split its documents into strata by `stratify_by` column
    (e.g. genre or author), then sample up to `nr_docs` per topic by
    allocating roughly equal samples per stratum and weighting by topic‐
    probability.

    This implementation guarantees at most `nr_docs` per topic, and fills up to
    `nr_docs` if possible by topping up from the highest-probability remaining docs.
    Ordering is deterministic for a given seed and increasing `nr_docs` always
    extends the previous selection.
    """
    repr_docs: dict[int, list[str]] = {}
    metadata_len = len(metadata)
    for tid in set(topics):
        idxs = [i for i, t in enumerate(topics) if t == tid]
        if not idxs:
            repr_docs[tid] = []
            continue

        strata: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            if i >= metadata_len:
                continue
            val = (
                metadata.get_column(stratify_by)[i]
                if stratify_by in metadata.columns
                else ""
            )
            strata[val].append(i)

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        base = nr_docs // max(1, len(strata))
        extra = nr_docs % max(1, len(strata))
        selected: list[int] = []

        # Phase 1: Even allocation across strata
        for stratum, members in strata.items():
            k = base + (1 if extra > 0 else 0)
            if extra > 0:
                extra -= 1
            weights = probs[members, tid] if probs.ndim == 2 else np.ones(len(members))
            if weights.sum() > 0:
                p = weights / weights.sum()
            else:
                p = None
            take = min(k, len(members))
            if take > 0:
                chosen = rng.choice(members, size=take, replace=False, p=p)
                selected.extend(chosen.tolist())

        # Phase 2: Top-up if we have less than nr_docs
        if len(selected) < nr_docs:
            remaining_pool = [
                i for i in idxs if i not in selected and 0 <= i < len(docs)
            ]
            if remaining_pool:
                weights = (
                    probs[remaining_pool, tid]
                    if probs.ndim == 2
                    else np.ones(len(remaining_pool))
                )
                order = np.argsort(weights)[::-1]  # highest probability first
                for idx in order:
                    selected.append(remaining_pool[idx])
                    if len(selected) >= nr_docs:
                        break

        # Final safety: clamp to valid doc indices
        max_idx = len(docs) - 1
        selected = [i for i in selected if 0 <= i <= max_idx]
        # Order is deterministic for a given seed
        selected.sort()
        repr_docs[tid] = [docs[i] for i in selected[:nr_docs]]
    return repr_docs


def calculate_model(
    docs: list[str],
    vectorizer: CountVectorizer,
    embedding_model: str,
    representation_model: list[str],
    prompt: str | None,
    nr_topics: str | int,
    device: str,
    language: str,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[BERTopic, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Create and fit a new BERTopic model with specified configuration.

    Handles model initialization, embedding computation, topic modeling,
    and dimensionality reduction. Supports various representation models
    including local transformers and API-based LLMs.

    Args:
        docs: Documents to fit the model on.
        vectorizer: Configured CountVectorizer for term extraction.
        embedding_model: Sentence transformer model name/path.
        representation_model: List of representation model names.
        prompt: Prompt template for LLM representations.
        nr_topics: Target number of topics (0/"auto" for automatic).
        device: Compute device specification.
        language: Document language for LLM prompts.
        model_kwargs: Optional model configuration (dtype, attention).

    Returns:
        Tuple of:
            - BERTopic: Fitted topic model
            - np.ndarray: Document embeddings
            - np.ndarray: 2D UMAP embeddings
            - list[int]: Topic assignments
            - np.ndarray: Topic probabilities

    Device specifications:
        - "cpu": CPU computation
        - "cuda:N": GPU N with default precision
        - "cuda:N-bnb": GPU N with 8-bit quantization

    Examples:
        >>> model, emb, umap_emb, topics, probs = calculate_model(
        ...     docs=["text1", "text2"],
        ...     vectorizer=my_vectorizer,
        ...     embedding_model="all-MiniLM-L6-v2",
        ...     representation_model=["KeyBERTInspired"],
        ...     device="cuda:0"
        ... )
    """
    dev_opt = device
    if device.endswith("-bnb"):
        dev_opt = device.removesuffix("-bnb")

    vectorizer = validate_vectorizer(vectorizer, language)

    # Handle MPS device with fallback
    if dev_opt == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            dev_opt = "cpu"

    # Fail early if there are no documents to model
    if not docs:
        raise ValueError(
            "No documents provided to calculate_model. Ensure corpus creation succeeded and text files are present."
        )

    representation_models = {}
    if representation_model:
        representation_models["Main"] = KeyBERTInspired()
        for name in representation_model:
            try:
                rm = build_representation_model(
                    name, language, prompt, device, model_kwargs
                )
                if rm:
                    representation_models[name] = rm
            except Exception as e:
                print(f"Failed to add representation model {name}: {e}")
                continue

    # Create SentenceTransformer with proper GPU/MPS configuration
    model_kwargs_dict = {
        "torch_dtype": torch.bfloat16
        if dev_opt.startswith("cuda") or dev_opt == "mps"
        else torch.float32,
        "attn_implementation": "flash_attention_2"
        if dev_opt.startswith("cuda")
        else None,
    }

    with log_time(
        f"sentence_transformer_init:{embedding_model}", extra={"device": dev_opt}
    ):
        try:
            sentence_embedding_model = SentenceTransformer(
                embedding_model,
                device=dev_opt,
                model_kwargs=model_kwargs_dict,
            )
        except ValueError as e:
            msg = str(e)
            # catch flash‐attention support error and retry without it
            if "does not support Flash Attention" in msg and model_kwargs_dict.get(
                "attn_implementation"
            ):
                logging.warning(
                    f"{embedding_model}: Flash Attention unsupported, retrying without it"
                )
                fallback_kwargs = {**model_kwargs_dict}
                fallback_kwargs.pop("attn_implementation", None)
                sentence_embedding_model = SentenceTransformer(
                    embedding_model,
                    device=dev_opt,
                    model_kwargs=fallback_kwargs,
                )
            else:
                raise

    with log_time("bertopic_init", extra={"nr_topics": nr_topics}):
        topic_model = BERTopic(
            embedding_model=sentence_embedding_model,
            vectorizer_model=vectorizer,
            representation_model=representation_models
            if representation_model
            else None,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )

    with log_time("bertopic_fit", extra={"n_docs": len(docs)}):
        topic_model = topic_model.fit(docs)

    with log_time("bertopic_transform", extra={"n_docs": len(docs)}):
        topics, probs = topic_model.transform(docs)

    with log_time(
        "embedding.encode", extra={"model": embedding_model, "n_docs": len(docs)}
    ):
        if isinstance(embedding_model, str) and "ruri" in embedding_model:
            embeddings = sentence_embedding_model.encode(
                docs,
                show_progress_bar=True,
                prompt="トピック: ",
            )
        else:
            embeddings = sentence_embedding_model.encode(
                docs,
                show_progress_bar=True,
            )

    with log_time("umap_reduction", extra={"n_neighbors": 10, "n_components": 2}):
        umap_model = get_umap_model(dev_opt)
        reduced_embeddings = umap_model.fit_transform(embeddings)

    with log_time("rep_models_cleanup"):
        rep_models = getattr(topic_model, "representation_model", None)
        if isinstance(rep_models, dict):
            for rm in rep_models.values():
                if isinstance(rm, CleanTextGeneration) and hasattr(rm, "_orig_model"):
                    try:
                        del rm._orig_model
                    except Exception:
                        pass

    with log_time("clear_device_cache", extra={"device": device}):
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        elif device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return topic_model, embeddings, reduced_embeddings, topics, probs


def save_computed(
    path, vectorizer_model, topics, probs, embeddings, reduced_embeddings
):
    with open(f"{path}-vectorizer_model_vocabulary.pickle", "wb") as f:
        pickle.dump(vectorizer_model.vocabulary_, f)
    with open(f"{path}-topics.pickle", "wb") as f:
        pickle.dump(topics, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-probs.pickle", "wb") as f:
        pickle.dump(probs, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-reduced_embeddings.pickle", "wb") as f:
        pickle.dump(reduced_embeddings, f, pickle.HIGHEST_PROTOCOL)
    Path(f"{path}.finished").touch()


# As long as this function does not do any unnecessary work, it should be the same as @st.cache or better
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
    max_df: float,
    tokenizer_features: list[str],
    tokenizer_pos_filter: set[str] | None,
    surface_filter: str | re.Pattern | None = None,
    topic_model: BERTopic | None = None,
    embeddings: np.ndarray | None = None,
    reduced_embeddings: np.ndarray | None = None,
    topics: list[int] | None = None,
    probs: np.ndarray | None = None,
    device: str = "cpu",
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Path, BERTopic, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Load cached BERTopic model or compute and cache a new one.

    This function implements a caching mechanism for BERTopic models to avoid
    recomputation. Models are identified by a hash of their configuration and
    input documents. Thread-safe file locking ensures concurrent access safety.

    Args:
        docs: List of documents to model.
        language: Language of documents ("Japanese" or "English").
        embedding_model: Name/path of sentence transformer model.
        representation_model: List of representation models to use:
            - "KeyBERTInspired", "MaximalMarginalRelevance"
            - "local:model/path" for local transformers models
            - "litellm:provider/model" for LiteLLM API models
        prompt: Custom prompt template for LLM-based representations.
        nr_topics: Number of topics to reduce to (0 or "auto" for automatic).
        tokenizer_type: Tokenizer engine for vectorization.
        dictionary_type: Dictionary for the tokenizer.
        ngram_range: Range of n-grams to extract.
        max_df: Maximum document frequency for terms.
        tokenizer_features: Token features to extract.
        tokenizer_pos_filter: POS tags to filter out.
        topic_model: Pre-existing model to save (skips computation).
        embeddings: Pre-computed document embeddings.
        reduced_embeddings: Pre-computed 2D embeddings.
        topics: Pre-computed topic assignments.
        probs: Pre-computed topic probabilities.
        device: Compute device ("cpu", "cuda:0", "cuda:0-bnb" for 8-bit).
        model_kwargs: Additional kwargs for embedding model (dtype, attention).

    Returns:
        Tuple of:
            - Path: Cache file path for the model
            - BERTopic: Loaded or computed topic model
            - np.ndarray: Document embeddings
            - np.ndarray: 2D reduced embeddings
            - list[int]: Topic assignments per document
            - np.ndarray: Topic probabilities per document

    Examples:
        >>> # First call computes and caches
        >>> path, model, emb, red_emb, topics, probs = load_and_persist_model(
        ...     docs=["doc1", "doc2"],
        ...     language="English",
        ...     embedding_model="all-MiniLM-L6-v2",
        ...     representation_model=["KeyBERTInspired"],
        ...     # ... other params
        ... )

        >>> # Subsequent calls with same params load from cache
        >>> path2, model2, *_ = load_and_persist_model(
        ...     docs=["doc1", "doc2"],
        ...     # ... same params
        ... )
        >>> assert path == path2  # Same cache file
    """
    docs_hash = xxhash.xxh3_64_hexdigest(pickle.dumps(docs, pickle.HIGHEST_PROTOCOL))
    settings_hash = xxhash.xxh3_64_hexdigest(
        pickle.dumps(
            (
                embedding_model,
                tuple(sorted(representation_model))
                if isinstance(representation_model, list)
                else representation_model,
                prompt,
                nr_topics,
                tokenizer_type,
                dictionary_type,
                ngram_range,
                max_df,
                tokenizer_features,
                tokenizer_pos_filter,
                surface_filter,
            ),
            pickle.HIGHEST_PROTOCOL,
        )
    )
    filename = f"{docs_hash}-{settings_hash}"
    path = Path(f"cache/{filename}")
    lock_path = Path(f"{path}.lock")

    # 1) clean up any really old lock file
    # Clean stale locks, then acquire and either load or compute under one lock
    _cleanup_stale_lock(lock_path)
    finished = path.with_suffix(".finished")
    lock = FileLock(str(lock_path), timeout=30)
    # Short random jitter to spread simultaneous acquires
    time.sleep(random.uniform(0.05, 0.25))
    with lock:
        # 1) if no model passed in AND we've completed once, load everything
        if topic_model is None and finished.exists():
            with log_time("load_from_cache", extra={"path": str(path)}):
                topic_model = BERTopic.load(str(path), embedding_model=embedding_model)
                # restore vectorizer vocabulary
                with open(f"{path}-vectorizer_model_vocabulary.pickle", "rb") as f:
                    v = pickle.load(f)
                topic_model.vectorizer_model = LanguageProcessor(
                    tokenizer_type=tokenizer_type,
                    dictionary_type=dictionary_type,
                    ngram_range=ngram_range,
                    max_df=max_df,
                    features=tokenizer_features,
                    pos_filter=tokenizer_pos_filter,
                    language=language,
                    vocabulary=v,
                ).vectorizer
                # unpickle topics/probs/embeddings/reduced_embeddings
                with open(f"{path}-topics.pickle", "rb") as f:
                    topics = pickle.load(f)
                with open(f"{path}-probs.pickle", "rb") as f:
                    probs = pickle.load(f)
                with open(f"{path}-embeddings.pickle", "rb") as f:
                    embeddings = pickle.load(f)
                with open(f"{path}-reduced_embeddings.pickle", "rb") as f:
                    reduced_embeddings = pickle.load(f)
                # Load pre-computed topics per class if available
                try:
                    topics_per_class_path = Path(
                        f"{path}-topics_per_class_genre.pickle"
                    )
                    if topics_per_class_path.exists():
                        with open(topics_per_class_path, "rb") as f:
                            topic_model.topics_per_class_genre_ = pickle.load(f)
                    else:
                        topic_model.topics_per_class_genre_ = None
                except Exception:
                    topic_model.topics_per_class_genre_ = None
                # Restore persisted topic aspects and custom labels if present
                try:
                    aspects_path = Path(f"{path}-topic_aspects.pickle")
                    labels_path = Path(f"{path}-custom_labels.pickle")
                    if aspects_path.exists():
                        with open(aspects_path, "rb") as f:
                            topic_model.topic_aspects_ = pickle.load(f)
                    if labels_path.exists():
                        with open(labels_path, "rb") as f:
                            topic_model.custom_labels_ = pickle.load(f)
                        try:
                            topic_model.set_topic_labels(topic_model.custom_labels_)
                        except Exception:
                            pass
                except Exception:
                    # Safe to continue without persisted labels/aspects
                    pass
                assert embeddings is not None
                assert reduced_embeddings is not None
                assert topics is not None
                assert probs is not None
                return path, topic_model, embeddings, reduced_embeddings, topics, probs

        # 2) otherwise compute new or re-save passed-in model
        if topic_model is None:
            logging.warning(f"Computing model {path}")
            with log_time(
                "compute_model",
                extra={"path": str(path), "nr_topics": nr_topics, "language": language},
            ):
                topic_model, embeddings, reduced_embeddings, topics, probs = (
                    calculate_model(
                        docs,
                        LanguageProcessor(
                            language=language,
                            tokenizer_type=tokenizer_type,
                            dictionary_type=dictionary_type,
                            ngram_range=ngram_range,
                            max_df=max_df,
                            features=tokenizer_features,
                            pos_filter=tokenizer_pos_filter,
                            surface_filter=surface_filter,
                        ).vectorizer,
                        embedding_model,
                        representation_model,
                        prompt,
                        nr_topics,
                        device,
                        language,
                        model_kwargs=model_kwargs,
                    )
                )

            # Pre-compute topics per class for genre stratification
            logging.info("Pre-computing topics per class for genre...")
            try:
                # Recreate metadata to get genre information
                all_metadata = get_metadata(language)
                if all_metadata is not None:
                    _, metadata = create_corpus_core(
                        all_metadata,
                        language,
                        # Use the same chunking parameters that were used for docs
                        chunksize=100,  # This should match the actual chunksize used
                        min_chunksize=20,  # This should match the actual min_chunksize used
                        chunks=50,  # This should match the actual chunks used
                        tokenizer_type=tokenizer_type,
                        dictionary_type=dictionary_type,
                        tokenizer_factory=lambda **kw: LanguageProcessor(**kw),
                    )
                    if not metadata.is_empty() and "genre" in metadata.columns:
                        genre_classes = metadata.get_column("genre").to_list()
                        topics_per_class_genre = topic_model.topics_per_class(
                            docs, genre_classes
                        )

                        # Store on the model for later access
                        topic_model.topics_per_class_genre_ = topics_per_class_genre
            except Exception as e:
                logging.warning(f"Failed to pre-compute topics per class: {e}")
                topic_model.topics_per_class_genre_ = None

            # persist everything under a timing block
            with log_time("persist_model", extra={"path": str(path)}):
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
                try:
                    topics_per_class_data = getattr(
                        topic_model, "topics_per_class_genre_", None
                    )
                    if topics_per_class_data is not None:
                        with open(f"{path}-topics_per_class_genre.pickle", "wb") as f:
                            pickle.dump(
                                topics_per_class_data, f, pickle.HIGHEST_PROTOCOL
                            )
                except Exception as e:
                    logging.warning(f"Failed to persist topics per class: {e}")

            finished.touch()
            with log_time("persist_aspects_labels"):
                try:
                    with open(f"{path}-topic_aspects.pickle", "wb") as f:
                        pickle.dump(
                            getattr(topic_model, "topic_aspects_", {}),
                            f,
                            pickle.HIGHEST_PROTOCOL,
                        )
                    with open(f"{path}-custom_labels.pickle", "wb") as f:
                        pickle.dump(
                            getattr(topic_model, "custom_labels_", None),
                            f,
                            pickle.HIGHEST_PROTOCOL,
                        )
                except Exception:
                    pass

            assert embeddings is not None
            assert reduced_embeddings is not None
            assert topics is not None
            assert probs is not None
            return path, topic_model, embeddings, reduced_embeddings, topics, probs
        else:
            # passed-in model → just extract topics & probs
            topics = topic_model.topics_
            probs = topic_model.probabilities_

        # 3) persist everything and mark finished
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

        # Persist pre-computed topics per class
        try:
            topics_per_class_data = getattr(
                topic_model, "topics_per_class_genre_", None
            )
            if topics_per_class_data is not None:
                with open(f"{path}-topics_per_class_genre.pickle", "wb") as f:
                    pickle.dump(topics_per_class_data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logging.warning(f"Failed to persist topics per class: {e}")

        finished.touch()
        # Persist topic aspects and custom labels to sidecar files
        try:
            with open(f"{path}-topic_aspects.pickle", "wb") as f:
                pickle.dump(
                    getattr(topic_model, "topic_aspects_", {}),
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
            with open(f"{path}-custom_labels.pickle", "wb") as f:
                pickle.dump(
                    getattr(topic_model, "custom_labels_", None),
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass
        assert embeddings is not None
        assert reduced_embeddings is not None
        assert topics is not None
        assert probs is not None
        return path, topic_model, embeddings, reduced_embeddings, topics, probs
