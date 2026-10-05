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
        "model",
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
        transformers_utils.load_transformers_model("model", device="cuda:0-bnb")


def test_five_users_share_one_embedding_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loads = []
    ready = Barrier(5)

    def model_loader(model_name, **kwargs):
        loads.append(model_name)
        return object()

    monkeypatch.setattr(transformers_utils, "SentenceTransformer", model_loader)

    def load(user):
        ready.wait(timeout=5)
        return transformers_utils.load_embedding_model("shared", "cpu")

    with ThreadPoolExecutor(max_workers=5) as pool:
        models = list(pool.map(load, range(5)))

    assert loads == ["shared"]
    assert all(model is models[0] for model in models)
