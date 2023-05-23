import tqdm
import stqdm

tqdm.orignal_class = stqdm.stqdm


import streamlit as st

st.set_page_config(layout="wide")

from data_lib import *
from gensim_lib import *
import re
from io import StringIO


st.title("Gensim (LDA) を使用したトピックモデル")

st.sidebar.markdown("## 設定")

# Custom HTML:
# st.components.v1.html(html.data, scrolling=True)

from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Set training parameters.
c01, c02 = st.sidebar.columns(2)
with c01:
    random_state = st.sidebar.number_input("Random state", min_value=0, value=42)
    num_topics = st.sidebar.number_input("Topics", min_value=2, value=40)
    iterations = st.sidebar.number_input("Iterations", min_value=1, value=2000)
with c02:
    chunksize = st.sidebar.number_input("Chunksize", min_value=10, value=4000)
    passes = st.sidebar.number_input("Passes", min_value=1, value=80)
eval_every = None  # Don't evaluate model perplexity, takes too much time.

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4d/Smoothed_LDA.png")
st.sidebar.markdown(
    """
-   $M$ denotes the number of documents
-   $K$ denotes the number of topics
-   $N$ is number of words in a given document (document $i$ has $N_{i}$ words)
-   $\\alpha$ is the parameter of the Dirichlet prior on the per-document topic distributions
-   $\\beta$ is the parameter of the Dirichlet prior on the per-topic word distribution
-   $\\theta_{i}$ is the topic distribution for document $i$
-   $\\varphi_{k}$ is the word distribution for topic $k$
-   $z_{ij}$ is the topic for the $j$-th word in document $i$
-   $w_{ij}$ is the specific word.
"""
)

all_metadata = get_metadata()
metadata, labels, authors, docs, original_docs = create_chunked_data(all_metadata)

st.plotly_chart(chunksizes_by_author_plot(metadata))


@st.cache_resource
def create_dictionary(_docs):
    dictionary = Dictionary(_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    return dictionary


@st.cache_resource
def create_gensim_corpus(_dic, docs):
    corpus = [_dic.doc2bow(doc) for doc in docs]
    return corpus


dictionary = create_dictionary(docs)
corpus = create_gensim_corpus(dictionary, docs)

from gensim.models.callbacks import Callback

# lda_callback = Callback(metrics=[])


@st.cache_data
def create_lda_model(
    _dic,
    corpus,
    random_state,
    num_topics,
    chunksize,
    passes,
    iterations,
    eval_every,  # Don't evaluate model perplexity, takes too much time.
):
    _temp = _dic[0]  # This is only to "load" the dictionary.
    id2word = _dic.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha="auto",
        eta="auto",
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every,
        random_state=random_state,  # 再現できるようにrandom seedを任意な数値に設定
        # callbacks=[],
    )
    return model


model = create_lda_model(
    dictionary,
    corpus,
    random_state=random_state,
    num_topics=num_topics,
    chunksize=chunksize,
    passes=passes,
    iterations=iterations,
    eval_every=eval_every,
)


from gensim.test.utils import datapath

export = st.checkbox("モデルファイルのエキスポート", value=False)

if export:
    temp_file = datapath("model")
    model.save(temp_file)
    st.download_button("トピックモデルファイルのダウンロード", temp_file)


st.markdown("## トピックモデル")

st.markdown("### 各トピックにおけるトーケン分布")

# print(model.show_topics(num_topics=model.num_topics, formatted=False))
st.write(
    pd.DataFrame(
        {
            topic_id: [f"{token}*{prob:.2f}" for token, prob in tokens]
            for topic_id, tokens in model.show_topics(
                num_topics=model.num_topics, formatted=False
            )
        }
    ).T
)

# st.write([
#     (topic, token, prob)
#     for topic_tokens, score in model.top_topics(corpus, topn=10)
#     for token, prob in topic_tokens])

# st.write(create_topic_df(model))

st.markdown("## 文章トピック行列 (Document-Topic Matrix)")

dtm = create_dtm(model, corpus, labels, metadata)

st.markdown("## トピック分布による作家の距離とクラスタ")

st.plotly_chart(create_dtm_heatmap(dtm))
# Look at how this is done in BERTopic:
# st.plotly_chart(create_dendrogram_heatmap(dtm))

st.markdown("## PyLDAvisによる可視化")

pyldavis_str = pyldavis_html(model, corpus, dictionary)

st.components.v1.html(pyldavis_str, width=1250, height=875, scrolling=True)

st.markdown("## 文章の可視化")

selection_col1, selection_col2 = st.columns(2)

with selection_col1:
    selected_author = st.selectbox("著者", options=set(authors))
author_works = set(
    metadata[metadata.authors == selected_author]["labels"].apply(
        lambda s: re.sub(r"_\d+$", ".txt", s)
    )
)
with selection_col2:
    selected_work = st.selectbox(
        "作品", options=all_metadata[all_metadata.filename.isin(author_works)]["title_ja"]
    )
selected_filename = all_metadata[all_metadata.title_ja == selected_work][
    "filename"
].apply(lambda s: s.replace(".txt", ""))
selected_label = st.select_slider(
    "作品チャンク",
    options=metadata[
        metadata.apply(
            lambda r: r["labels"].startswith(selected_filename.iloc[0]), axis=1
        )
    ]["labels"],
)

docid = metadata.query(f"labels == '{selected_label}'")["docid"].to_list()[0]

st.components.v1.html(
    colorize_topics(docid, dictionary, model, corpus, docs, original_docs, labels),
    height=500,
    scrolling=True,
)

st.markdown("## テキストファイルのトピック推定")

uploaded_file = st.file_uploader("ファイルをアップロード")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()

    tagger = get_tagger()
    file_tokens = tokenize(string_data, tagger)
    file_bow = dictionary.doc2bow(file_tokens)

    st.components.v1.html(
        colorize_topics(
            0,
            dictionary,
            model,
            [file_bow],
            [file_tokens],
            [file_tokens],
            [uploaded_file.name],
        ),
        height=500,
        scrolling=True,
    )
