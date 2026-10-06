import csv
import io
import os
import shutil
from ipaddress import ip_address, ip_network
from pathlib import Path

import streamlit as st

MAX_TEXT_CHARS = 20_000
MAX_PROMPT_CHARS = 4_000
MAX_LABEL_CHARS = 200
MAX_UPLOAD_BYTES = 100_000
MAX_DOCUMENTS = 50_000
MAX_CACHE_BYTES = 8 * 1024**3
UNIVERSITY_NETWORK = ip_network("133.1.0.0/16")
CLIENT_IP_HEADER = "X-Topic-Client-IP"


def full_access() -> bool:
    if os.environ.get("TOPIC_MODELING_PUBLIC") != "1":
        return True
    try:
        return (
            ip_address(st.context.headers.get(CLIENT_IP_HEADER, ""))
            in UNIVERSITY_NETWORK
        )
    except ValueError:
        return False


def require_compute_access() -> None:
    if not full_access():
        raise PermissionError(
            "Computation is available from university addresses only."
        )


def validate_text(text: str, limit: int = MAX_TEXT_CHARS) -> None:
    if len(text) > limit:
        raise ValueError(f"Text must not exceed {limit:,} characters.")


def validate_upload(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Files must not exceed {MAX_UPLOAD_BYTES:,} bytes.")


def check_cache_capacity(additional_bytes: int = 0) -> None:
    directory = Path("cache")
    directory.mkdir(exist_ok=True)
    size = 0
    for path in directory.rglob("*"):
        if path.is_file():
            try:
                size += path.stat().st_size
            except FileNotFoundError:
                continue
    if (
        size + additional_bytes > MAX_CACHE_BYTES
        or shutil.disk_usage(directory).free < additional_bytes + 1024**3
    ):
        raise ValueError(
            "The model cache is full. Ask the administrator to remove unused models."
        )


def csv_bytes(table, *, index: bool = False) -> bytes:
    def escape(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ([escape(table.index.name or "")] if index else [])
        + [escape(name) for name in table.columns]
    )
    for key, row in zip(table.index, table.itertuples(index=False, name=None)):
        writer.writerow(
            ([escape(key)] if index else []) + [escape(value) for value in row]
        )
    return output.getvalue().encode("utf-8")
