from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from filelock import FileLock


def model_cache_lock(lock_path: Path) -> FileLock:
    return FileLock(str(lock_path), timeout=-1)


@cache
def _device_lock(lock_path: Path) -> FileLock:
    return FileLock(str(lock_path), timeout=-1)


@contextmanager
def device_compute_lock(device: str) -> Iterator[None]:
    lock_name = device.removesuffix("-bnb").replace(":", "_")
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _device_lock((cache_dir / f"compute-{lock_name}.lock").resolve()):
        yield
