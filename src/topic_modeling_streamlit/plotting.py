import altair as alt
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from bertopic import BERTopic
from plotly.graph_objs._figure import Figure  # type: ignore[import-not-found]

from topic_modeling_streamlit.utils import get_umap_model, strip_topic_prefix


def build_topic_label_map(
    topic_model: BERTopic, custom_labels: bool | str = False
) -> dict[int, str]:
    """
    Build a mapping from topic id -> display label for a BERTopic instance.
    Prefers CustomName (if present), otherwise Name, and strips any leading
    "{tid}: " prefix so labels are independent of list indices.
    """
    info = topic_model.get_topic_info().reset_index(drop=True)
    label_map: dict[int, str] = {}

    for r in info.to_dict(orient="records"):
        tid = int(r["Topic"])
        name = r.get("CustomName") or r.get("Name") or ""
        # Using helper to strip consistent prefix
        clean_name = (
            strip_topic_prefix([name], [tid])[0] if isinstance(name, str) else name
        )
        label_map[tid] = clean_name or str(tid)
    return label_map


def chunksizes_by_author_plot(metadata: pl.DataFrame) -> Figure:
    """
    Box plot of document length per author.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({'author': ['a','a','b'], 'length': [1,2,3], 'label': ['l1','l2','l3'], 'docid': [0,1,2]})
        >>> fig = chunksizes_by_author_plot(df)
        >>> hasattr(fig, 'to_dict')
        True
    """
    return px.box(
        metadata.to_pandas(),
        x="author",
        y="length",
        hover_data=["label", "docid"],
        title="Author document length (characters) distribution",
    )


def visualize_documents_altair(
    docs: list[str],
    topic_model: BERTopic,
    embeddings: np.ndarray | None = None,
    reduced_embeddings: np.ndarray | None = None,
    sample: float = 1.0,
    topics_per_doc: list[int] | None = None,
    label_map: dict[int, str] | None = None,
    debug_return_df: bool = False,
    custom_labels: bool | str = False,
    hide_annotations: bool = False,
    width: int = 800,
    height: int = 600,
    authors: list[str] | None = None,
    works: list[str] | None = None,
    chunk_ids: list[int] | None = None,
    color_by: str = "Topic",
    selectable: bool = False,
    show_outliers: bool = True,
    device: str = "cpu",
) -> alt.Chart:
    """
    Create an interactive 2D scatter plot of documents using UMAP embeddings.

    Documents are positioned based on their semantic similarity and colored by
    topic assignment, authorship, or work. The visualization helps identify
    topic clusters and document relationships.

    Args:
        docs: List of document texts to visualize.
        topic_model: Fitted BERTopic model containing topic assignments.
        embeddings: High-dimensional document embeddings. If None, will be
            extracted from documents using the topic model.
        reduced_embeddings: Pre-computed 2D UMAP embeddings. If None, will
            be computed from embeddings.
        sample: Fraction of documents to sample per topic (0.0 to 1.0).
            Helps manage large datasets while maintaining topic balance.
        topics: Specific topic IDs to highlight. If None, all topics shown.
            Non-selected topics will be colored as "Other".
        custom_labels: Whether to use custom topic labels. Can be:
            - False: Use default topic ID labels
            - True: Use model's custom labels if available
            - str: Name of specific topic aspect to use for labels
        hide_annotations: If True, hide topic label annotations on plot.
        width: Chart width in pixels.
        height: Chart height in pixels.
        authors: Optional list of document authors (same length as docs).
        works: Optional list of work titles (same length as docs).
        chunk_ids: Optional list of chunk IDs (same length as docs).
        color_by: Field to use for coloring points. Options:
            - "Topic": Color by topic assignment
            - "Author": Color by document author
            - "Work": Color by work title
        selectable: If True, enable click selection on points (requires chunk_ids).

    Returns:
        Altair Chart object with interactive 2D visualization.

    Examples:
        >>> # Basic usage with pre-computed embeddings
        >>> chart = visualize_documents_altair(
        ...     docs=["Document 1", "Document 2"],
        ...     topic_model=fitted_model,
        ...     reduced_embeddings=umap_embeddings
        ... )

        >>> # Color by author with custom labels
        >>> chart = visualize_documents_altair(
        ...     docs=documents,
        ...     topic_model=model,
        ...     authors=author_list,
        ...     color_by="Author",
        ...     custom_labels=True
        ... )
    """
    # Use the provided per-document topics array if given; otherwise fall back to the model's internal topics_
    if topics_per_doc is None:
        topics_per_doc = list(getattr(topic_model, "topics_", []))
    # Ensure an integer list for sampling logic
    topics_per_doc = [int(t) for t in topics_per_doc]

    # 2) clip sample to [0,1]
    sample = min(max(sample, 0.0), 1.0)

    # 3) sample indices per topic
    selected: list[int] = []
    for t in set(topics_per_doc):
        idxs = np.where(np.array(topics_per_doc) == t)[0]
        k = max(1, int(len(idxs) * sample))
        selected.extend(np.random.choice(idxs, k, replace=False).tolist())
    # FIXME Clamp selected to valid doc indices
    max_len = len(docs)
    selected = [i for i in selected if 0 <= i < max_len]
    if not selected:
        # no valid indices left; bail out with empty chart
        return alt.Chart(pl.DataFrame({"x": [], "y": [], "label": []}))
    selected_arr = np.array(selected, dtype=int)

    # 4) get or compute 2D embeddings
    if reduced_embeddings is not None:
        emb2d = reduced_embeddings[selected_arr]
    else:
        # Prepare source embeddings
        if embeddings is not None:
            emb_source = embeddings[selected_arr]
        else:
            emb_source = topic_model._extract_embeddings(
                [docs[i] for i in selected], method="document"
            )

        # Try GPU UMAP via torchdr

        umap_model = get_umap_model(device or "cpu")
        emb2d = umap_model.fit_transform(emb_source)

    # 5) build a Polars DataFrame
    data = {
        "x": emb2d[:, 0].tolist(),
        "y": emb2d[:, 1].tolist(),
        "topic": np.array(topics_per_doc)[selected_arr].tolist(),
        "doc": [docs[i] for i in selected],
    }
    if authors is not None:
        data["author"] = [authors[i] for i in selected]
    if works is not None:
        data["work"] = [works[i] for i in selected]
    if chunk_ids is not None:
        data["chunk_id"] = np.array(chunk_ids)[selected_arr].tolist()
    df = pl.DataFrame(data)

    # Optionally drop outlier topic -1
    if not show_outliers:
        df = df.filter(pl.col("topic") != -1)

    # 6) select which topics to highlight (only for topic-color mode)
    unique = sorted(df["topic"].unique().to_list())
    topics_to_highlight = unique
    df = df.with_columns(
        pl.when(pl.col("topic").is_in(topics_to_highlight))
        .then(pl.col("topic"))
        .otherwise(-1)
        .alias("topic")
    )

    # 7) map topic -> label using an explicit mapping keyed by topic id
    if label_map is None:
        label_map = build_topic_label_map(topic_model, custom_labels=custom_labels)
    # Guarantee an 'Other' entry for outliers / unknowns
    label_map = {**{-1: "Other"}, **{int(k): v for k, v in label_map.items()}}
    # Populate the df 'label' column from topic ids
    df = df.with_columns(
        pl.col("topic")
        .map_elements(lambda x: label_map.get(int(x), "Unknown"), return_dtype=pl.Utf8)
        .alias("label")
    )

    # 8) pick grouping field and compute centroids in Polars
    field_map = {"Topic": "label", "Author": "author", "Work": "work"}
    group_field = field_map.get(color_by, "label")
    df = df.with_columns(pl.col(group_field).alias("color_group"))
    centroids = (
        df.group_by("color_group")
        .agg(
            [
                pl.col("x").mean().alias("x"),
                pl.col("y").mean().alias("y"),
            ]
        )
        .with_columns(pl.col("color_group").alias("label"))
    )

    # 9) zero-axes lines
    zero = pl.DataFrame({"coord": ["x", "y"], "val": [0.0, 0.0]})

    # 10) build Altair layers
    tooltip_fields = ["label:N", "doc:N"]
    if "author" in df.columns:
        tooltip_fields.append("author:N")
    if "work" in df.columns:
        tooltip_fields.append("work:N")
    if "chunk_id" in df.columns:
        tooltip_fields.append("chunk_id:Q")

    # Determine theme and stroke color for label outlines
    theme = st.context.theme.type  # "light" or "dark"
    stroke_color = "black" if theme == "dark" else "white"

    points = (
        alt.Chart(df)
        .mark_circle(size=20, opacity=0.5)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.condition(
                alt.datum.color_group == "Other",
                alt.value("lightgray"),
                alt.Color("color_group:N", legend=None),
            ),
            tooltip=tooltip_fields,
        )
    )
    vline = (
        alt.Chart(pl.DataFrame(zero.filter(pl.col("coord") == "x")))
        .mark_rule(color="lightgray")
        .encode(x="val:Q")
    )
    hline = (
        alt.Chart(pl.DataFrame(zero.filter(pl.col("coord") == "y")))
        .mark_rule(color="lightgray")
        .encode(y="val:Q")
    )

    layers = [points, vline, hline]
    if not hide_annotations:
        # Theme-aware outline color for labels
        labels_layer = (
            alt.Chart(centroids)
            .mark_text(
                dx=5,
                dy=-5,
                fontSize=12,
                fontWeight="bold",
                stroke=stroke_color,
                strokeWidth=0.5,
            )
            .encode(
                x="x:Q",
                y="y:Q",
                text="label:N",
                # Use same field as points for color so scale is shared
                color=alt.Color("color_group:N", legend=None),
            )
        )
        layers.insert(1, labels_layer)

    # base chart layer
    chart = alt.layer(*layers)

    # optionally add click-to-select on chunk_id
    if selectable and chunk_ids is not None:
        point_sel = alt.selection_point(
            fields=["chunk_id"],
            name="point_selection",
            on="click",
        )
        chart = chart.add_params(point_sel)

    chart = chart.properties(width=width, height=height).interactive()

    # style according to Streamlit light/dark mode
    if theme == "dark":
        chart = (
            chart.configure_view(stroke=None)
            .configure(background="#0E1117")
            .configure_axis(
                domainColor="white",
                gridColor="#444",
                labelColor="white",
                titleColor="white",
            )
            .configure_legend(labelColor="white", titleColor="white")
            .configure_text()
        )
    else:
        chart = (
            chart.configure_view(stroke=None)
            .configure(background="white")
            .configure_axis(
                domainColor="black",
                gridColor="#EEE",
                labelColor="black",
                titleColor="black",
            )
            .configure_legend(labelColor="black", titleColor="black")
            .configure_text()
        )

    if debug_return_df:
        return chart, df
    return chart
