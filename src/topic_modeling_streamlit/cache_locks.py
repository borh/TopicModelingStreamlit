from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from threading import BoundedSemaphore, local

from filelock import FileLock, Timeout

from topic_modeling_streamlit.security import require_compute_access

_slots = BoundedSemaphore(8)
_request = local()


@contextmanager
def _admission() -> Iterator[None]:
    require_compute_access()
    depth = getattr(_request, "depth", 0)
    if depth == 0 and not _slots.acquire(blocking=False):
        raise ValueError(
            "All eight computation slots are occupied. Please retry shortly."
        )
    _request.depth = depth + 1
    try:
        yield
    except Timeout as exc:
        raise ValueError("Computation is busy. Please retry shortly.") from exc
    finally:
        _request.depth = depth
        if depth == 0:
            _slots.release()


@contextmanager
def model_cache_lock(lock_path: Path) -> Iterator[None]:
    with _admission(), FileLock(str(lock_path), timeout=600):
        yield


@cache
def _device_lock(lock_path: Path) -> FileLock:
    return FileLock(str(lock_path), timeout=600)


@contextmanager
def device_compute_lock(device: str) -> Iterator[None]:
    lock_name = device.removesuffix("-bnb").replace(":", "_")
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (
        _admission(),
        _device_lock((cache_dir / f"compute-{lock_name}.lock").resolve()),
    ):
        yield
