from abc import ABC, abstractmethod
import re
import logging
import pysbd
import spacy
import rhoknp
from fugashi import Tagger, GenericTagger, create_feature_wrapper
import sudachipy
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import (
    # TextGeneration,
    # BaseRepresentation,
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
)
import bertopic.representation as representation
import tiktoken
from openai import AzureOpenAI
from os import environ as env
from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer

CustomFeatures = create_feature_wrapper(
    "CustomFeatures",
    (
        "pos1 pos2 pos3 pos4 cType cForm lForm lemma orth pron "
        "pronBase goshu orthBase iType iForm fType fForm "
        "kana kanaBase form formBase iConType fConType aType "
        "aConType lid lemma_id"
    ).split(" "),
)


class Dictionary(ABC):
    def __init__(self, name: str):
        self.name = name


class SpacyDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class FugashiDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class SudachipyDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class JumanppDictionary(Dictionary):
    def __init__(self, name: str):
        super().__init__(name)


class Tokenizer(ABC):
    def __init__(
        self,
        dictionary: Dictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        self.dictionary = dictionary
        self.segmenter = segmenter
        self.features = features
        self.pos_filter = pos_filter
        self.whitespace_rx = whitespace_rx

    @abstractmethod
    def get_features(self, t) -> str:
        pass

    @abstractmethod
    def tokenize(self, s: str) -> list[str]:
        pass


class SpacyTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: SpacyDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        self.model = spacy.load(self.dictionary.name)

    def get_features(self, t) -> str:
        mapping = {
            "orth": lambda t: t.orth_,
            "pos1": lambda t: t.pos_,
            "lemma": lambda t: t.lemma_,
            "text_with_ws": lambda t: t.text_with_ws,
        }
        if not self.features:
            return t.surface
        else:
            return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        sentences = self.segmenter.segment(s)
        return [
            self.get_features(t)
            for doc in self.model.pipe(sentences, disable=["parser", "ner"])
            for t in doc
            if not self.whitespace_rx.match(t.orth_)
            and not (self.pos_filter and t.pos_ in self.pos_filter)
        ]


class FugashiTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        if "65_novel" in dictionary.name:
            self.model = GenericTagger(
                dictionary.name,
                wrapper=CustomFeatures,
            )
        else:
            self.model = Tagger(dictionary.name)

    def get_features(self, t) -> str:
        if not self.features:
            return t.surface

        mapping = {
            "orth": lambda t: t.surface,
            "pos1": lambda t: t.feature.pos1,
            # Unknown/OOV words will get surface form returned
            "lemma": lambda t: t.feature.lemma or t.surface,
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence)
            if not self.whitespace_rx.match(t.surface)
            and not (self.pos_filter and t.feature.pos1 in self.pos_filter)
        ]


class SudachipyTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        match dictionary.name[-1]:
            case "A":
                self.mode = sudachipy.SplitMode.A
            case "B":
                self.mode = sudachipy.SplitMode.B
            case "C":
                self.mode = sudachipy.SplitMode.C

        sudachi = sudachipy.Dictionary(dict="full").create()  # TODO
        self.model = sudachi.tokenize

    def get_features(self, t) -> str:
        if not self.features:
            return t.surface()
        mapping = {
            "orth": lambda t: t.surface(),
            "pos1": lambda t: t.part_of_speech()[0],
            "lemma": lambda t: t.normalized_form(),
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence, self.mode)
            if not self.whitespace_rx.match(t.surface())
            and not (self.pos_filter and t.part_of_speech()[0] not in self.pos_filter)
        ]


class JumanppTokenizer(Tokenizer):
    def __init__(
        self,
        dictionary: FugashiDictionary,
        segmenter: pysbd.Segmenter,
        features: list[str] | None,
        pos_filter: set[str] | None,
        whitespace_rx: re.Pattern = re.compile(r"^\s*$"),
    ):
        super().__init__(dictionary, segmenter, features, pos_filter, whitespace_rx)
        self.model = rhoknp.Jumanpp().apply_to_sentence

    def get_features(self, t) -> str:
        if not self.features:
            return t.text

        mapping = {
            "orth": lambda t: t.text,
            "pos1": lambda t: t.pos,
            "lemma": lambda t: t.lemma,
        }
        return "/".join(mapping[feature](t) for feature in self.features)

    def tokenize(self, s: str) -> list[str]:
        return [
            self.get_features(t)
            for sentence in self.segmenter.segment(s)
            for t in self.model(sentence).morphemes
            if not self.whitespace_rx.match(t.text)
            and not (self.pos_filter and t.pos in self.pos_filter)
        ]


TOKENIZER_MAP = {
    "spaCy": SpacyTokenizer,
    "MeCab": FugashiTokenizer,
    "Sudachi": SudachipyTokenizer,
    "Juman++": JumanppTokenizer,
}

DEVENV_DICDIR = ".devenv/profile/share/mecab/dic"

DICTIONARY_MAP = {
    "近現代口語小説UniDic": FugashiDictionary("-r ./65_novel/dicrc -d ./65_novel/"),
    "UniDic-CWJ": FugashiDictionary(f"-d {DEVENV_DICDIR}/unidic-cwj"),
    "SudachiDict-full/A": SudachipyDictionary("SudachiDict-full/A"),
    "SudachiDict-full/B": SudachipyDictionary("SudachiDict-full/B"),
    "SudachiDict-full/C": SudachipyDictionary("SudachiDict-full/C"),
    "ja_core_news_sm": SpacyDictionary("ja_core_news_sm"),
    "Jumandict": JumanppDictionary(""),
    "en_core_web_sm": SpacyDictionary("en_core_web_sm"),
    "en_core_web_trf": SpacyDictionary("en_core_web_trf"),
}


class LanguageProcessor:
    def __init__(
        self,
        language: str = "Japanese",
        tokenizer_type: str = "MeCab",
        dictionary_type: str = "UniDic-CWJ",
        ngram_range: tuple[int, int] = (1, 1),
        features: list[str] | None = None,
        pos_filter: set[str] | None = None,
        vocabulary: dict[str, int] | None = None,
    ):
        self.language = language
        self.segmenter = pysbd.Segmenter(
            language=self.language[:2].lower(), clean=False
        )
        self.whitespace_rx: re.Pattern = re.compile(r"^\s*$")  # TODO

        self.features = features
        self.pos_filter = pos_filter

        # Pickle helpers:
        self.tokenizer_type = tokenizer_type
        self.dictionary_type = dictionary_type

        self.dictionary = DICTIONARY_MAP[dictionary_type]
        self.tokenizer = TOKENIZER_MAP[tokenizer_type](
            self.dictionary,
            self.segmenter,
            self.features,
            self.pos_filter,
            self.whitespace_rx,
        )

        self.ngram_range = ngram_range

        self.vocabulary = vocabulary

        if self.vocabulary is not None:
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                tokenizer=self.tokenizer.tokenize,
                lowercase=False if language == "Japanese" else True,
                token_pattern=r".+",  # i.e. no pattern; take all
                vocabulary=vocabulary,
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                tokenizer=self.tokenizer.tokenize,
                lowercase=False if language == "Japanese" else True,
                token_pattern=r".+",  # i.e. no pattern; take all
            )

        if self.language == "English":
            assert isinstance(self.tokenizer, SpacyTokenizer)
            assert isinstance(self.dictionary, SpacyDictionary)
        elif self.language == "Japanese":
            assert isinstance(
                self.tokenizer,
                (
                    FugashiTokenizer,
                    SudachipyTokenizer,
                    SpacyTokenizer,
                    JumanppTokenizer,
                ),
            )
            assert isinstance(
                self.dictionary,
                (
                    FugashiDictionary,
                    SudachipyDictionary,
                    SpacyDictionary,
                    JumanppDictionary,
                ),
            )

    def __repr__(self):
        return f"JapaneseTagger<{self.tokenizer_type},{self.dictionary_type},features={self.features},pos_filter={self.pos_filter}>"

    def __getstate__(self):
        logging.error("getstate")
        return (
            self.language,
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.features,
            self.pos_filter,
            self.vocabulary,
        )

    def __setstate__(self, state):
        logging.error("setstate")
        (
            self.language,
            self.tokenizer_type,
            self.dictionary_type,
            self.ngram_range,
            self.features,
            self.pos_filter,
            self.vocabulary,
        ) = state
        # Restore from initial params
        self.__init__(
            language=self.language,
            tokenizer_type=self.tokenizer_type,
            dictionary_type=self.dictionary_type,
            ngram_range=self.ngram_range,
            features=self.features,
            pos_filter=self.pos_filter,
            vocabulary=self.vocabulary,
        )


#         # Limit maximum sentence length to a reasonable amount (around 2000)
#                 token_splits = split(sentence, 2000)
#                 tokens.extend(
#                     self._sudachi_features_fn(t)
#                     for ts in token_splits
#                     for t in self.tagger(ts, self.mode)
#                     if not self.whitespace_rx.match(t.surface())
#                     and not (
#                         self.pos_filter and not t.part_of_speech()[0] in self.pos_filter
#                     )
#                 )
#         return tokens

# def __reduce__(self):
#    logging.error("reduce")
#    return (
#        type(self),
#        (
#            self.tokenizer_type,
#            self.dictionary_type,
#            self.ngram_range,
#            self._mecab_tokenizer_fn,
#            self._sudachi_tokenizer_fn,
#            self._jumanpp_tokenizer_fn,
#        ),
#    )

# def __reduce_ex__(self, protocol):
#     logging.error("reduce_ex")
#     return (
#         type(self),
#         (
#             self.tokenizer_type,
#             self.dictionary_type,
#             self.ngram_range,
#             self._mecab_tokenizer_fn,
#             self._sudachi_tokenizer_fn,
#             self._jumanpp_tokenizer_fn,
#         ),
#     )

# def tokenize(self, s):
#     sentences = self.sentence_segmenter.segment(s)
#     return [t for sentence in sentences for t in self.tokenizer_fn(sentence)]


import cloudpickle
import pickle
from pathlib import Path


# @st.cache_resource
# def get_openai():
#     with open("/run/agenix/openai-api") as f:
#         openai.api_key = f.read().strip()
#     representation_model = OpenAI(model="gpt-4", delay_in_seconds=10, chat=True)
#     return representation_model


def calculate_model(
    docs: list[str],
    vectorizer: CountVectorizer,
    embedding_model: str,
    representation_model: list[str],
    prompt: str | None,
    nr_topics: str | int,
    device: str,
):
    if representation_model:
        representation_models = {}

        representation_models["Main"] = KeyBERTInspired()
        if "KeyBERTInspired" in representation_model:
            representation_models["KeyBERTInspired"] = KeyBERTInspired()
        if "MaximalMarginalRelevance" in representation_model:
            representation_models["MaximalMarginalRelevance"] = (
                MaximalMarginalRelevance(diversity=0.5)
            )
        if "GPT-4-0613" in representation_model:
            tokenizer = tiktoken.encoding_for_model("gpt-4-0613")
            client = AzureOpenAI(
                # model_name="gpt-4-0613",
                api_key=env["AZURE_API_KEY"],
                api_version=env["AZURE_API_VERSION"],
                azure_endpoint=env["AZURE_API_BASE"],
                azure_deployment=env["AZURE_DEPLOYMENT_NAME"],
                # config=OpenAIConfig(seed=42, temperature=0.0),
            )
            representation_model = representation.OpenAI(
                client,
                # model="gpt-4-0613",
                chat=True,
                prompt="""Q:
I have created a topic model and have a topic that contains the following documents in Japanese:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the above information, can you give a short label of the topic of at most 5 words in Japanese?
A:
""",
                delay_in_seconds=2,
                nr_docs=5,
                doc_length=80,
                tokenizer=tokenizer,
                generator_kwargs={"seed": 42, "temperature": 0.2},
            )
            representation_models["gpt-4-0613"] = representation_model

        # Only with spaCy:
        # elif representation_model == "PartOfSpeech":
        #     representation_model = representation.PartOfSpeech()
        # elif representation_model and "/" in representation_model:
        #     if not prompt:
        #         prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
        #     generator = pipeline("text2text-generation", model=representation_model)
        #     representation_model = TextGeneration(generator, prompt=prompt)
        # else:
        #     raise Exception(f"Unsupported representation model {representation_model}")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            representation_model=representation_models,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )
    else:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            verbose=True,
            calculate_probabilities=True,
            nr_topics=nr_topics,
        )

    sentence_model = SentenceTransformer(embedding_model, device=device)
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    topic_model = topic_model.fit(docs, embeddings)
    topics, probs = topic_model.transform(docs, embeddings)
    reduced_embeddings = UMAP(
        n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
    ).fit_transform(embeddings)

    return topic_model, embeddings, reduced_embeddings, topics, probs


def save_computed(
    path, vectorizer_model, topics, probs, embeddings, reduced_embeddings
):
    with open(f"{path}-vectorizer_model_vocabulary.pickle", "wb") as f:
        pickle.dump(vectorizer_model.vocabulary_, f)
    with open(f"{path}-topics.pickle", "wb") as f:
        pickle.dump(topics, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-probs.pickle", "wb") as f:
        pickle.dump(probs, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{path}-reduced_embeddings.pickle", "wb") as f:
        pickle.dump(reduced_embeddings, f, pickle.HIGHEST_PROTOCOL)
    Path(f"{path}.finished").touch()


from filelock import FileLock
from time import sleep
import xxhash


# As long as this function does not do any unnecessary work, it should be the same as @st.cache or better
def load_and_persist_model(
    docs: list[str],
    language: str,
    embedding_model: str,
    representation_model: list[str],
    prompt: str | None,
    nr_topics: str | int,
    tokenizer_type: str,
    dictionary_type: str,
    ngram_range: tuple[int, int],
    tokenizer_features: list[str],
    tokenizer_pos_filter: set[str] | None,
    topic_model=None,
    embeddings=None,
    reduced_embeddings=None,
    topics=None,
    probs=None,
    device: str = "cpu",
):
    docs_hash = xxhash.xxh3_64_hexdigest(pickle.dumps(docs, pickle.HIGHEST_PROTOCOL))
    settings_hash = xxhash.xxh3_64_hexdigest(
        pickle.dumps(
            (
                embedding_model,
                representation_model,
                prompt,
                nr_topics,
                tokenizer_type,
                dictionary_type,
                ngram_range,
                tokenizer_features,
                tokenizer_pos_filter,
            ),
            pickle.HIGHEST_PROTOCOL,
        )
    )
    # representation_id = ""
    # if representation_model:
    #     representation_id = f"{'_'.join(k.replace('/', '_') for k in representation_model)}-{xxhash.xxh3_64_hexdigest(prompt) if prompt else ''}-"
    # filename = f"{docs_hash}-{tokenizer_type}-{dictionary_type.replace('/', '_')}-{tokenizer_features}-{tokenizer_pos_filter}-{embedding_model.replace('/', '_')}-{representation_id}{nr_topics}"
    filename = f"{docs_hash}-{settings_hash}"
    path = Path(f"cache/{filename}")
    lock_path = Path(f"{path}.lock")
    logging.info(f"Lock not yet created at {lock_path}. Assuming first writer.")
    if not lock_path.exists():
        lock = FileLock(lock_path)
        logging.warning(f"Computation not started. Getting {lock}")
        with lock.acquire(poll_interval=1):
            if topic_model:
                logging.warning(f"Saving computed model {topic_model} to {path}")
                topics = topic_model.topics_
                probs = topic_model.probabilities_
                # topic_model.vectorizer_model = None # Prevent pickling of vectorizer
                topic_model.save(
                    path,
                    serialization="safetensors",
                    save_embedding_model=embedding_model,
                    save_ctfidf=True,
                )
                save_computed(
                    path,
                    topic_model.vectorizer_model,
                    topics,
                    probs,
                    embeddings,
                    reduced_embeddings,
                )
            else:
                logging.warning(f"Computing model {path}")
                (
                    topic_model,
                    embeddings,
                    reduced_embeddings,
                    topics,
                    probs,
                ) = calculate_model(
                    docs,
                    LanguageProcessor(
                        tokenizer_type=tokenizer_type,
                        dictionary_type=dictionary_type,
                        ngram_range=ngram_range,
                        features=tokenizer_features,
                        pos_filter=tokenizer_pos_filter,
                        language=language,
                    ).vectorizer,
                    embedding_model,
                    representation_model,
                    prompt,
                    nr_topics,
                    device,
                )
                logging.warning(f"Saving model {path}")
                # topic_model.vectorizer_model = None # Prevent pickling of vectorizer
                topic_model.save(
                    path,
                    serialization="safetensors",
                    save_embedding_model=embedding_model,
                    save_ctfidf=True,
                )
                save_computed(
                    path,
                    topic_model.vectorizer_model,
                    topics,
                    probs,
                    embeddings,
                    reduced_embeddings,
                )
    elif not topic_model:
        chech_finished = Path(f"{path}.finished")
        logging.warning(f"Loading precomputed model {path}")
        if chech_finished.exists():
            topic_model = BERTopic.load(str(path), embedding_model=embedding_model)
            # # Currently not restored correctly:
            # # https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_save_utils.py#L334
            # We save and load only the vocabulary to prevent native code pickle errors.
            with open(f"{path}-vectorizer_model_vocabulary.pickle", "rb") as f:
                v = pickle.load(f)
                topic_model.vectorizer_model = LanguageProcessor(
                    tokenizer_type=tokenizer_type,
                    dictionary_type=dictionary_type,
                    ngram_range=ngram_range,
                    features=tokenizer_features,
                    pos_filter=tokenizer_pos_filter,
                    language=language,
                    vocabulary=v,
                ).vectorizer

            with open(f"{path}-topics.pickle", "rb") as f:
                topics = pickle.load(f)
            with open(f"{path}-probs.pickle", "rb") as f:
                probs = pickle.load(f)
            with open(f"{path}-embeddings.pickle", "rb") as f:
                embeddings = pickle.load(f)
            with open(f"{path}-reduced_embeddings.pickle", "rb") as f:
                reduced_embeddings = pickle.load(f)
        else:
            logging.error(
                f"Failed to access completed save (checked: {chech_finished}), sleeping for 1 second and retrying..."
            )
            sleep(1)
            return load_and_persist_model(
                docs,
                language,
                embedding_model,
                representation_model,
                prompt,
                nr_topics,
                tokenizer_type,
                dictionary_type,
                ngram_range,
                tokenizer_features,
                tokenizer_pos_filter,
                topic_model=topic_model,
                embeddings=embeddings,
                reduced_embeddings=reduced_embeddings,
                topics=topics,
                probs=probs,
                device=device,
            )
    else:  # Do nothing
        logging.warning("Noop: model already cached and loaded")
        if not topics:
            topics = topic_model.topics_
        if not probs:
            probs = topic_model.probabilites_
        assert topics is not None
        assert probs is not None
        assert embeddings is not None
        assert reduced_embeddings is not None

    logging.warning("Model loading complete.")

    return path, topic_model, embeddings, reduced_embeddings, topics, probs
