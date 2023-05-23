import streamlit as st
import pandas as pd
from fugashi import GenericTagger


# @st.cache_resource
# _pickle.PicklingError: Can't pickle <function japanese_tokenizer at 0x7fc725705b20>: it's not the same object as __main__.japanese_tokenizer
# MeCab is not picklable:
# https://github.com/polm/fugashi/issues/54
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


# tokenize("考察する。かなりかかつた。僕は考える")
# tokenize("考察する。かなりかかつた。僕は考える", lemma=True)
# tokenize("考察する。かなりかかつた。太郎は大阪城公園で考える。重本さんもそうした玉藻", lemma=True, remove_proper_nouns=True)


@st.cache_data
def get_metadata():
    metadata = pd.read_csv(
        "Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv", sep="\t"
    )
    return metadata


import plotly.express as px


def chunksizes_by_author_plot(metadata):
    # df = px.data.tips()
    fig = px.box(
        metadata, x="authors", y="lengths", points="all", hover_data=["labels", "docid"]
    )
    return fig
