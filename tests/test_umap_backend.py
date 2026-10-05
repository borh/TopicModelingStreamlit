import pytest
import torch
import torchdr

from topic_modeling_streamlit.utils import get_umap_model


@pytest.mark.parametrize("device", ["cuda:1", "cuda:1-bnb"])
def test_gpu_umap_does_not_require_faiss_and_preserves_device(monkeypatch, device):
    parameters = {}

    def umap(**kwargs):
        parameters.update(kwargs)
        return object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torchdr, "UMAP", umap)
    get_umap_model(device)

    assert parameters["backend"] is None
    assert parameters["device"] == "cuda:1"
