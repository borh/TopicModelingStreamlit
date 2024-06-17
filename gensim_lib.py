import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


import streamlit as st
import pandas as pd
import polars as pl
from gensim.models import LdaModel

# top_topics = model.top_topics(corpus)

from pathlib import Path
from fugashi import Tagger


# # @st.cache_resource
# # _pickle.PicklingError: Can't pickle <function japanese_tokenizer at 0x7fc725705b20>: it's not the same object as __main__.japanese_tokenizer
# # MeCab is not picklable:
# # https://github.com/polm/fugashi/issues/54
# def get_tagger(flags="-r ./65_novel/dicrc -d ./65_novel/"):
#     tagger = GenericTagger(flags)
#     return tagger
#
#
# def get_lemma(token):
#     if len(token.feature) > 6:
#         return token.feature[7]
#     else:  # 未知語
#         return token.surface
#
#
# def tokenize(text, tagger, lemma=False, remove_proper_nouns=False):
#     tokens = []
#     for token in tagger(text):
#         if remove_proper_nouns and token.feature[1] == "固有名詞":
#             tokens.append("")
#             continue
#         if token == "<EOS>":
#             continue
#
#         tokens.append(get_lemma(token) if lemma else token.surface)
#     return tokens
#
#
@st.cache_data
def get_metadata():
    metadata_df = pl.read_csv(
        "Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv", separator="\t"
    )
    return metadata_df


DEVENV_DICDIR = ".devenv/profile/share/mecab/dic"


@st.cache_resource
def get_tagger(
    flags=f"-d {DEVENV_DICDIR}/unidic-novel -r {DEVENV_DICDIR}/unidic-novel/dicrc",
):
    tagger = Tagger(flags)
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

        author = (
            metadata.filter(pl.col("filename") == Path(file).name)
            .select(pl.col("author_ja"))
            .head(1)[0]
        )

        current_chunk = 0
        for sentence in sentences:
            tokens = tokenize(
                sentence, tagger, lemma=lemma, remove_proper_nouns=remove_proper_nouns
            )
            chunks[current_chunk].extend(tokens)
            if len(chunks[current_chunk]) >= chunk_size:
                labels.append(filename + "_" + str(current_chunk))
                # authors.append(
                #     metadata.query(f'filename == "{Path(file).name}"')
                #     .iloc[0, :]
                #     .author_ja
                # )
                authors.append(author)
                current_chunk += 1
                chunks.append([])
        if len(chunks[-1]) < chunk_size:
            chunks.pop()
        return labels, authors, chunks


@st.cache_data
def create_chunked_data(_all_metadata, chunksize=2000, tagger=get_tagger()):
    labels = []
    filenames = []
    authors = []
    docs = []
    original_docs = []

    for file in Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob(
        "*.txt"
    ):
        chunk_labels, chunk_authors, chunks = chunk_tokens(
            file,
            tagger,
            _all_metadata,
            lemma=True,
            remove_proper_nouns=True,
            chunk_size=chunksize,
        )

        labels.extend(chunk_labels)
        filenames.extend([file.name for _ in range(len(chunk_labels))])
        authors.extend(chunk_authors)
        docs.extend(chunks)

        _, _, original_chunks = chunk_tokens(
            file, tagger, _all_metadata, lemma=False, chunk_size=chunksize
        )
        original_docs.extend(original_chunks)
    metadata = pl.DataFrame(
        {
            "author": authors,
            "label": labels,
            "filename": filenames,
            "length": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )
    metadata = _all_metadata.select(
        pl.col("filename", "genre", "year"),
        pl.col("author_ja").alias("author"),
        pl.col("title_ja").alias("title"),
    ).join(
        metadata.select(pl.col("filename", "label", "docid", "length")), on="filename"
    )

    return metadata, docs, original_docs


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


def topic2dense(topic_probs, num_topics):
    d = {topic: prob for topic, prob in topic_probs}
    return [d[i] if i in d else 0.0 for i in range(num_topics)]


@st.cache_data
def create_dtm(_model, corpus, authors, collapsed=True):
    dt = [topic2dense(_model.get_document_topics(d), _model.num_topics) for d in corpus]
    dtm = pd.DataFrame(dt, columns=range(_model.num_topics), index=authors)
    if collapsed:
        return dtm.groupby(level=0).mean()
    else:
        return dtm


def create_dtm_heatmap(dtm):
    return px.imshow(dtm, text_auto=".2f", aspect="auto")


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
