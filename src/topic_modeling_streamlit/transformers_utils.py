"""
Unified transformers model loading utilities.

Provides consistent model loading with device handling, quantization,
and configuration across the codebase.
"""

import logging
import os
from typing import Any, Literal

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from topic_modeling_streamlit.cache_locks import device_compute_lock
from topic_modeling_streamlit.model_revisions import model_identity
from topic_modeling_streamlit.security import require_compute_access

logger = logging.getLogger(__name__)


def get_attention_implementation(device: str) -> str | None:
    if not device.startswith("cuda"):
        return None
    return "sdpa"


def load_transformers_model(
    model_name: str,
    device: str = "cpu",
    task: Literal["text-generation", "text2text-generation"] = "text-generation",
    max_new_tokens: int = 20,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    require_compute_access()
    model_identity(model_name)
    if trust_remote_code:
        raise ValueError("Remote model code is disabled.")
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("dtype", model_kwargs.get("torch_dtype", torch.bfloat16))
    attention = model_kwargs.get("attn_implementation")
    if attention == "flash_attention_2" and torch.version.hip:
        attention = "sdpa"
    model_kwargs["attn_implementation"] = attention or get_attention_implementation(
        device
    )
    with device_compute_lock(device):
        return _load_transformers_model(
            model_name, device, task, max_new_tokens, trust_remote_code, model_kwargs
        )


@st.cache_resource(show_spinner=False, max_entries=5, hash_funcs={torch.dtype: str})
def _load_transformers_model(
    model_name: str,
    device: str = "cpu",
    task: Literal["text-generation", "text2text-generation"] = "text-generation",
    max_new_tokens: int = 20,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    model_kwargs = model_kwargs or {}

    # Parse device and quantization
    use_bnb = device.endswith("-bnb")
    clean_device = device.removesuffix("-bnb")

    if use_bnb and torch.version.hip is not None:
        raise ValueError("8-bit BitsAndBytes quantization is unavailable on ROCm")

    # Handle MPS fallback
    if clean_device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        clean_device = "cpu"

    if (
        clean_device.startswith("cuda")
        and os.environ.get("TOPIC_MODELING_PUBLIC") == "1"
    ):
        torch.cuda.set_per_process_memory_fraction(0.5, clean_device)

    # Configure quantization (not supported on MPS)
    quantization_config = None
    if use_bnb and clean_device.startswith("cuda"):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif use_bnb and clean_device == "mps":
        print("Warning: Quantization not supported on MPS, using full precision")

    # Configure model loading
    model_config: dict[str, Any] = {
        "trust_remote_code": False,
        "revision": model_identity(model_name)[1],
        "use_safetensors": True,
        "weights_only": True,
        "device_map": {"": clean_device}
        if clean_device.startswith("cuda")
        else clean_device,
    }

    # Add dtype and attention for CUDA and MPS
    if clean_device.startswith("cuda"):
        attention_implementation = model_kwargs.get("attn_implementation")
        if attention_implementation == "flash_attention_2" and torch.version.hip:
            attention_implementation = "sdpa"
        elif attention_implementation is None:
            attention_implementation = get_attention_implementation(clean_device)
        model_config.update(
            {
                "dtype": model_kwargs["dtype"],
                "attn_implementation": attention_implementation,
                "quantization_config": quantization_config,
            }
        )
    elif clean_device == "mps":
        model_config.update(
            {
                "dtype": model_kwargs["dtype"],
                # Note: flash_attention_2 may not be supported on MPS
            }
        )
    else:
        model_config["quantization_config"] = quantization_config

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=False, revision=model_identity(model_name)[1]
    )

    # Create pipeline
    pipeline_kwargs: dict[str, Any] = {
        "task": task,
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": max_new_tokens,
    }

    pipe = pipeline(**pipeline_kwargs)

    return pipe, tokenizer


def load_embedding_model(model_name: str, device: str = "cpu") -> SentenceTransformer:
    require_compute_access()
    model_name, _ = model_identity(model_name)
    device = device.removesuffix("-bnb")
    with device_compute_lock(device):
        return _load_embedding_model(model_name, device)


@st.cache_resource(show_spinner=False, max_entries=5)
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    if device.startswith("cuda") and os.environ.get("TOPIC_MODELING_PUBLIC") == "1":
        torch.cuda.set_per_process_memory_fraction(0.5, device)
    model_kwargs = {
        "dtype": torch.bfloat16
        if device.startswith("cuda") or device == "mps"
        else torch.float32,
        "attn_implementation": get_attention_implementation(device),
        "use_safetensors": model_name != "AnnaWegmann/Style-Embedding",
        "weights_only": True,
    }
    try:
        return SentenceTransformer(
            model_name,
            device=device,
            revision=model_identity(model_name)[1],
            trust_remote_code=False,
            model_kwargs=model_kwargs,
        )
    except ValueError as e:
        if "does not support Flash Attention" not in str(e):
            raise
        logger.warning(
            "%s: Flash Attention unsupported, retrying without it", model_name
        )
        model_kwargs.pop("attn_implementation")
        return SentenceTransformer(
            model_name,
            device=device,
            revision=model_identity(model_name)[1],
            trust_remote_code=False,
            model_kwargs=model_kwargs,
        )
