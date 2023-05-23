import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


import streamlit as st
import pandas as pd
from gensim.models import LdaModel

# top_topics = model.top_topics(corpus)

from pathlib import Path
from fugashi import GenericTagger


@st.cache_resource
def get_tagger(flags="-r ./65_novel/dicrc -d ./65_novel/"):
    tagger = GenericTagger(flags)
    return tagger


def get_lemma(token):
    if len(token.feature) > 6:
        return token.feature[7]
    else:  # 未知語
        return token.surface


def tokenize(text, tagger, lemma=False, remove_proper_nouns=False):
    tokens = []
    for token in tagger(text):
        if remove_proper_nouns and token.feature[1] == "固有名詞":
            tokens.append("")
            continue
        if token == "<EOS>":
            continue

        tokens.append(get_lemma(token) if lemma else token.surface)
    return tokens


def chunk_tokens(
    file, tagger, metadata, lemma=False, remove_proper_nouns=False, chunk_size=2000
):
    """
    文章を約2,000形態素のチャンクに分割する。
    分割方法として，文を2つのチャンクに区分されないように，まとまった文をのみチャンクに割り当てる。
    そのため，それぞれのチャンクは2,000形態素以上になる。
    """

    with open(file, encoding="utf-8") as f:
        filename = Path(file).stem

        text = f.read()
        text = text.replace("。", "。\n")
        text = text.replace("\n\n", "\n")
        sentences = text.splitlines()

        labels = []
        authors = []
        chunks = [[]]
        current_chunk = 0
        for sentence in sentences:
            tokens = tokenize(
                sentence, tagger, lemma=lemma, remove_proper_nouns=remove_proper_nouns
            )
            chunks[current_chunk].extend(tokens)
            if len(chunks[current_chunk]) >= chunk_size:
                labels.append(filename + "_" + str(current_chunk))
                authors.append(
                    metadata.query(f'filename == "{Path(file).name}"')
                    .iloc[0, :]
                    .author_ja
                )
                current_chunk += 1
                chunks.append([])
        if len(chunks[-1]) < chunk_size:
            chunks.pop()
        return labels, authors, chunks


@st.cache_data
def create_chunked_data(metadata, chunksize=2000, tagger=get_tagger()):
    labels = []
    authors = []
    docs = []
    original_docs = []

    for file in Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob(
        "*.txt"
    ):
        chunk_labels, chunk_authors, chunks = chunk_tokens(
            file,
            tagger,
            metadata,
            lemma=True,
            remove_proper_nouns=True,
            chunk_size=chunksize,
        )

        labels.extend(chunk_labels)
        authors.extend(chunk_authors)
        docs.extend(chunks)

        _, _, original_chunks = chunk_tokens(
            file, tagger, metadata, lemma=False, chunk_size=chunksize
        )
        original_docs.extend(original_chunks)
    metadata = pd.DataFrame(
        {
            "authors": authors,
            "labels": labels,
            "lengths": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )
    return metadata, labels, authors, docs, original_docs


# avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
# print('Average topic coherence: %.4f.' % avg_topic_coherence)
#
# from pprint import pprint
# pprint(top_topics[:2])

import pyLDAvis
import pyLDAvis.gensim_models as gensim_models

# pyLDAvis.disable_notebook()


# @st.cache_resource
def pyldavis_html(_model, corpus, _dictionary) -> str:
    data = gensim_models.prepare(
        _model, corpus, _dictionary, start_index=0, sort_topics=False
    )
    html = pyLDAvis.prepared_data_to_html(data)
    return html


@st.cache_data
def create_topic_df(_model):
    ttm = pd.DataFrame(
        _model.get_topics(),
        columns=_model.id2word.values(),
        index=[f"{i}" for i in range(_model.num_topics)],
    )
    return ttm


# import plotly.figure_factory as ff
#
# fig = ff.create_dendrogram(ttm, orientation='left', distfun=cosine_similarity)
# fig.update_layout(width=800, height=1400)
# fig.show()
#
#
# import umap
#
# ttm_umap = umap.UMAP(n_components=2, metric='cosine').fit(ttm.T)
#
# ttm_df = pd.DataFrame(ttm_umap.embedding_, columns=["x", "y"], index=model.id2word.values())
#
# fig = px.scatter(ttm_df, x="x", y="y", labels=model.id2word.values(), text=model.id2word.values())
# fig.update_traces(textposition="top center")
# fig.show()


def topic2dense(topic_probs, num_topics):
    d = {topic: prob for topic, prob in topic_probs}
    return [d[i] if i in d else 0.0 for i in range(num_topics)]


@st.cache_data
def create_dtm(_model, corpus, labels, metadata, collapsed=True):
    dt = [topic2dense(_model.get_document_topics(d), _model.num_topics) for d in corpus]
    dtm = pd.DataFrame(dt, columns=range(_model.num_topics), index=metadata.authors)
    if collapsed:
        return dtm.groupby(level=0).mean()
    else:
        return dtm


def create_dtm_heatmap(dtm):
    return px.imshow(dtm, text_auto=".2f", aspect="auto")


import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from scipy.spatial.distance import pdist, squareform


def create_dendrogram_heatmap(data):
    # get data
    data_array = data.to_numpy()
    labels = data.index

    # Initialize figure by creating upper dendrogram
    # print(data_array)
    # print(labels)
    fig = ff.create_dendrogram(
        data_array, orientation="bottom", labels=labels, distfun=cosine_similarity
    )
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(
        data_array, orientation="right", distfun=cosine_similarity
    )
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
    dendro_leaves = list(map(int, dendro_leaves))
    # print(dendro_leaves)
    data_dist = pdist(data_array)
    # print(data_array.shape)
    # print(data_dist.shape)
    heat_data = squareform(data_dist)
    # print(heat_data.shape)
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]

    heat_data = np.triu(heat_data)

    heatmap = [
        go.Heatmap(x=dendro_leaves, y=dendro_leaves, z=heat_data, colorscale="Blues")
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout(
        {
            "width": 1000,
            "height": 1000,
            "showlegend": False,
            "hovermode": "closest",
        }
    )
    # Edit xaxis
    fig.update_layout(
        xaxis={
            "domain": [0.15, 1],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
        }
    )
    # Edit xaxis2
    fig.update_layout(
        xaxis2={
            "domain": [0, 0.15],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "domain": [0, 0.85],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )
    # Edit yaxis2
    fig.update_layout(
        yaxis2={
            "domain": [0.825, 0.975],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        }
    )

    # Plot!
    return fig


import plotly.express as px


from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from operator import itemgetter

from colorsys import rgb_to_hsv, hsv_to_rgb


def complementary(hex):
    """returns RGB components of complementary color"""
    r, g, b = mcolors.to_rgb(hex)
    hsv = rgb_to_hsv(r, g, b)
    return mcolors.to_hex(
        hsv_to_rgb((hsv[0] + 0.5) % 1, (hsv[1] - 0.5) % 1, (hsv[2] - 0.5) % 1)
    )


def invert_color(hex):
    rgb = mcolors.to_rgb(hex)
    inverted_rgb = (1 - rgb[0], 1 - rgb[1], 1 - rgb[2])
    return mcolors.to_hex(inverted_rgb)


def generate_topic_colormap(n):
    cm1 = plt.get_cmap("tab20b")
    cm2 = plt.get_cmap("tab20c")

    if n <= 10:
        return [[mcolors.rgb2hex(cm1(1.0 * i / n)), None] for i in range(n)]
    else:
        # n2 = n // 2
        return (
            [[mcolors.rgb2hex(cm1(1.0 * i / 20)), None] for i in range(20)]
            + [[mcolors.rgb2hex(cm2(1.0 * i / 20)), None] for i in range(20)]
            + [
                [
                    mcolors.rgb2hex(cm1(1.0 * i / 20)),
                    complementary(mcolors.rgb2hex(cm1(1.0 * i / 20))),
                ]
                for i in range(20)
            ]
            + [
                [
                    mcolors.rgb2hex(cm2(1.0 * i / 20)),
                    complementary(mcolors.rgb2hex(cm2(1.0 * i / 20))),
                ]
                for i in range(20)
            ]
        )


def colorize(s, fg, topicid=None, border=None):
    border = f"border:1px solid {border};" if border else ""
    token = (
        f"<ruby>{s}<rp>(</rp><rt>{topicid}</rt><rp>)</rp></ruby>" if topicid else f"{s}"
    )
    return f"<span style='color:{fg};background-color:white;{border}'>{token}</span>"


topic2color = generate_topic_colormap(40)  # FIXME


def show_topic_colors(model, num_topics):
    for topic, terms in model.show_topics(num_topics):
        c = topic2color[topic]
        display(
            HTML(colorize(str(topic) + ": " + terms, c[0], topicid=topic, border=c[1]))
        )


def colorize_topics(
    docid, dictionary, model, corpus, docs, original_docs, labels, assigned_only=True
):
    """
    docidに対応する文章の単語別トピック配属情報を色で表示する。トピックIDはルビで記述する。
    トピックは多い場合は，1文章のみに限定し，その文のトピックのみに色を配分することによって
    見やすくなるかもしれない。
    """
    sample = corpus[docid]
    inferred_topics = sorted(model[sample], key=itemgetter(1), reverse=True)
    inferred_topic_ids = {topic_id for topic_id, _ in inferred_topics}
    o = "<div style='background-color:white;'>"
    o += f"<p>{labels[docid]}: {inferred_topics}</p>"
    for position, token in enumerate(docs[docid]):
        if token in dictionary.token2id:
            top_token_topics = sorted(
                model.get_term_topics(
                    dictionary.token2id[token], minimum_probability=0.0
                ),
                key=itemgetter(1),
                reverse=True,
            )
            top_token_topic = [topic_id for topic_id, _prob in top_token_topics]
            if assigned_only:
                filtered_topics = [
                    topic_id
                    for topic_id in top_token_topic
                    if topic_id in inferred_topic_ids
                ]
                if filtered_topics:
                    top_token_topic = filtered_topics[0]
                    color = topic2color[top_token_topic]
                else:
                    top_token_topic = None
                    color = ["black", "white"]
            else:
                top_token_topic = top_token_topic[0]
                color = topic2color[top_token_topic]
        else:
            top_token_topic = None  # No assigned topics as OOV
            color = ["black", "white"]
        o += colorize(
            original_docs[docid][position], color[0], top_token_topic, color[1]
        )
    return o + "</div>"


# colorize_topics(21)

# model[dictionary.doc2bow(["朝日", "夜", "朝"])]
