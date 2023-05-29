import streamlit as st
import polars as pl
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


@st.cache_data
def get_metadata():
    metadata_df = pl.read_csv(
        "Aozora-Bunko-Fiction-Selection-2022-05-30/groups.csv", separator="\t"
    )
    return metadata_df
