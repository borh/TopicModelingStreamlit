import base64
import json
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from filelock import FileLock

from topic_modeling_streamlit.security import full_access, require_compute_access

MAX_ANALYSES = 20
MAX_ANALYSIS_BYTES = 10 * 1024**2


def publish_analysis(
    app: str, key: str, topics: bytes, downloads: dict[str, bytes]
) -> None:
    require_compute_access()
    if app not in {"bertopic", "gensim"} or not re.fullmatch(r"[a-f0-9]{16}", key):
        raise ValueError("Invalid analysis identifier.")
    if 4 * (len(topics) + sum(map(len, downloads.values()))) // 3 > MAX_ANALYSIS_BYTES:
        raise ValueError("The published analysis exceeds the 10 MB limit.")
    payload = json.dumps(
        {
            "topics": base64.b64encode(topics).decode(),
            "downloads": {
                name: base64.b64encode(data).decode()
                for name, data in downloads.items()
            },
        }
    ).encode()
    if len(payload) > MAX_ANALYSIS_BYTES:
        raise ValueError("The published analysis exceeds the 10 MB limit.")
    directory = Path("published") / app
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".lock"), timeout=10):
        path = directory / f"{key}.json"
        if not path.exists() and len(list(directory.glob("*.json"))) >= MAX_ANALYSES:
            raise ValueError(
                "The publication limit has been reached. Ask the administrator to remove unused analyses."
            )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
                temporary = Path(file.name)
                file.write(payload)
                os.fchmod(file.fileno(), 0o640)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def public_view(app: str) -> None:
    if full_access():
        return
    st.title(
        f"{'BERTopic' if app == 'bertopic' else 'Gensim (LDA)'} — published analyses"
    )
    st.info(
        "Full access is available from university addresses (133.1.0.0/16). Published analyses and downloads are available here."
    )
    directory = Path("published") / app
    paths = sorted(directory.glob("*.json"), key=lambda path: path.name)[:MAX_ANALYSES]
    if not paths:
        st.info("No analyses have been published yet.")
        st.stop()
    selected = st.selectbox("Saved analysis", paths, format_func=lambda path: path.stem)
    try:
        if selected.stat().st_size > MAX_ANALYSIS_BYTES:
            raise ValueError("Published analysis is too large.")
        data = json.loads(selected.read_bytes())
        st.dataframe(
            pd.read_csv(
                BytesIO(base64.b64decode(data["topics"], validate=True)),
                keep_default_na=False,
            ),
            hide_index=True,
        )
        for name, encoded in data["downloads"].items():
            st.download_button(
                f"Download {name}",
                base64.b64decode(encoded, validate=True),
                file_name=f"{app}-{name}.csv",
                mime="text/csv",
                on_click="ignore",
            )
    except (OSError, ValueError, KeyError):
        st.error("This published analysis is unavailable.")
    st.stop()


def publish_button(
    app: str, key: str, topics: bytes, downloads: dict[str, bytes]
) -> None:
    with st.expander("Publish analysis"):
        st.caption(
            "Publish the topic table and these downloads for anyone to view. Custom labels are included; inference text and prompts are excluded."
        )
        if st.button("Publish analysis", key=f"publish_{app}"):
            try:
                publish_analysis(app, key, topics, downloads)
                st.success("Analysis published.")
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    public_view(os.environ["TOPIC_MODELING_APP"])
