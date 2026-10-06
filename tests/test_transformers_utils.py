from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from topic_modeling_streamlit import transformers_utils
from topic_modeling_streamlit.utils import build_device_choices


@pytest.fixture(autouse=True)
def clear_model_resources():
    transformers_utils._load_transformers_model.clear()
    transformers_utils._load_embedding_model.clear()
    yield
    transformers_utils._load_transformers_model.clear()
    transformers_utils._load_embedding_model.clear()


def test_rocm_model_loading_uses_sdpa(monkeypatch):
    model_config = {}

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            model_config.update(kwargs)
            return object()

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            return object()

    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    monkeypatch.setattr(
        transformers_utils.AutoModelForCausalLM,
        "from_pretrained",
        ModelLoader.from_pretrained,
    )
    monkeypatch.setattr(
        transformers_utils.AutoTokenizer,
        "from_pretrained",
        TokenizerLoader.from_pretrained,
    )
    monkeypatch.setattr(transformers_utils, "pipeline", lambda **kwargs: object())

    transformers_utils.load_transformers_model(
        "Qwen/Qwen3-1.7B",
        device="cuda:0",
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )

    assert model_config["attn_implementation"] == "sdpa"
    assert model_config["device_map"] == {"": "cuda:0"}


def test_rocm_does_not_offer_bitsandbytes_devices(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert build_device_choices() == ["cuda:0", "cpu"]


def test_rocm_rejects_manual_bitsandbytes_device(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)

    with pytest.raises(ValueError, match="unavailable on ROCm"):
        transformers_utils.load_transformers_model(
            "Qwen/Qwen3-1.7B", device="cuda:0-bnb"
        )


def test_eight_users_share_one_embedding_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loads = []
    ready = Barrier(8)

    def model_loader(model_name, **kwargs):
        loads.append(model_name)
        return object()

    monkeypatch.setattr(transformers_utils, "SentenceTransformer", model_loader)

    def load(user):
        ready.wait(timeout=5)
        return transformers_utils.load_embedding_model("cl-nagoya/ruri-v3-30m", "cpu")

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(load, range(8)))

    assert loads == ["cl-nagoya/ruri-v3-30m"]
    assert all(model is models[0] for model in models)


def test_model_loading_rejects_unknown_models_and_remote_code():
    with pytest.raises(ValueError, match="supported model"):
        transformers_utils.load_embedding_model("unlisted/model")
    with pytest.raises(ValueError, match="Remote model code"):
        transformers_utils.load_transformers_model(
            "Qwen/Qwen3-1.7B", trust_remote_code=True
        )


def test_model_loading_uses_pinned_safe_weights(monkeypatch):
    from topic_modeling_streamlit.model_revisions import MODEL_REVISIONS

    configs: dict[str, dict] = {}
    monkeypatch.setattr(
        transformers_utils.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **kwargs: configs.update(model=kwargs),
    )
    monkeypatch.setattr(
        transformers_utils.AutoTokenizer,
        "from_pretrained",
        lambda name, **kwargs: configs.update(tokenizer=kwargs),
    )
    monkeypatch.setattr(transformers_utils, "pipeline", lambda **kwargs: object())
    transformers_utils.load_transformers_model("Qwen/Qwen3-1.7B")
    for config in configs.values():
        assert config["revision"] == MODEL_REVISIONS["Qwen/Qwen3-1.7B"]
        assert config["trust_remote_code"] is False
    assert configs["model"]["use_safetensors"] is True
    assert configs["model"]["weights_only"] is True


def test_public_model_loading_is_denied_even_with_warm_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loads = []
    monkeypatch.setattr(
        transformers_utils,
        "SentenceTransformer",
        lambda name, **kwargs: loads.append(name),
    )
    transformers_utils.load_embedding_model("cl-nagoya/ruri-v3-30m")
    monkeypatch.setenv("TOPIC_MODELING_PUBLIC", "1")
    with pytest.raises(PermissionError):
        transformers_utils.load_embedding_model("cl-nagoya/ruri-v3-30m")
    assert len(loads) == 1
