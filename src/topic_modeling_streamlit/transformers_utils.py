"""
Unified transformers model loading utilities.

Provides consistent model loading with device handling, quantization,
and configuration across the codebase.
"""

from typing import Any, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def load_transformers_model(
    model_name: str,
    device: str = "cpu",
    task: Literal["text-generation", "text2text-generation"] = "text-generation",
    max_new_tokens: int = 20,
    trust_remote_code: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """
    Load a transformers model with consistent device and quantization handling.

    Args:
        model_name: HuggingFace model name or path
        device: Device specification ("cpu", "cuda:0", "cuda:0-bnb")
        task: Pipeline task type
        max_new_tokens: Maximum new tokens for generation
        trust_remote_code: Allow custom model code
        model_kwargs: Additional model configuration

    Returns:
        Tuple of (pipeline, tokenizer)
    """
    model_kwargs = model_kwargs or {}

    # Parse device and quantization
    use_bnb = device.endswith("-bnb")
    clean_device = device.removesuffix("-bnb")

    # Handle MPS fallback
    if clean_device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        clean_device = "cpu"

    # Configure quantization (not supported on MPS)
    quantization_config = None
    if use_bnb and clean_device.startswith("cuda"):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif use_bnb and clean_device == "mps":
        print("Warning: Quantization not supported on MPS, using full precision")

    # Configure model loading
    model_config = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if clean_device.startswith("cuda") else clean_device,
    }

    # Add dtype and attention for CUDA and MPS
    if clean_device.startswith("cuda"):
        model_config.update(
            {
                "torch_dtype": model_kwargs.get("torch_dtype", torch.bfloat16),
                "attn_implementation": model_kwargs.get(
                    "attn_implementation", "flash_attention_2"
                ),
                "quantization_config": quantization_config,
            }
        )
    elif clean_device == "mps":
        model_config.update(
            {
                "torch_dtype": model_kwargs.get("torch_dtype", torch.bfloat16),
                # Note: flash_attention_2 may not be supported on MPS
            }
        )
    else:
        model_config["quantization_config"] = quantization_config

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Create pipeline
    pipeline_kwargs = {
        "task": task,
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": max_new_tokens,
    }

    # Only add device parameter if not using accelerate
    # Check if model was loaded with accelerate (device_map="auto" or quantization)
    using_accelerate = clean_device.startswith("cuda") and (
        model_config.get("device_map") == "auto" or quantization_config is not None
    )

    if not using_accelerate and clean_device.startswith("cuda"):
        _, idx = clean_device.split(":")
        pipeline_kwargs["device"] = int(idx)
    elif clean_device == "mps":
        # MPS device handling for pipeline
        pipeline_kwargs["device"] = clean_device

    pipe = pipeline(**pipeline_kwargs)

    return pipe, tokenizer
