MODEL_REVISIONS = {
    "cl-nagoya/ruri-v3-30m": "24899e5de370b56d179604a007c0d727bf144504",
    "sbintuitions/sarashina-embedding-v1-1b": "d060fcd8984075071e7fad81baff035cbb3b6c7e",
    "StyleDistance/mstyledistance": "d66ed25e48225a503b21a65bc804caf06c886f96",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "prdev/mini-gte": "15b421628a1084d99e24cec3fa0614c8f3dd9179",
    "AnnaWegmann/Style-Embedding": "d7d0f5ca829316a8f5695e49dfce80b86db5e76c",
    "sentence-transformers/all-MiniLM-L12-v2": "a50ef00143b4d5391434df20ae11632588ac25be",
    "sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "thenlper/gte-base": "c078288308d8dee004ab72c6191778064285ec0c",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}


def model_identity(name: str) -> tuple[str, str]:
    if "/" not in name:
        name = f"sentence-transformers/{name}"
    try:
        return name, MODEL_REVISIONS[name]
    except KeyError as exc:
        raise ValueError("Choose a supported model.") from exc
