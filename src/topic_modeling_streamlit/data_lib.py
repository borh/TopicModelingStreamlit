import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Callable, Protocol, TypedDict

import jaconv
import polars as pl

from topic_modeling_streamlit.gensim_lib import get_tagger, tokenize


class TokenizerFactory(Protocol):
    def __call__(self, **kwargs) -> object: ...
    @property
    def tokenizer(self) -> object: ...


KATAKANA = set(
    list(
        "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソ"
        "ゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペ"
        "ホボポマミムメモャヤュユョヨラリルレロワヲンーヮヰヱヵヶヴ"
    )
)

HIRAGANA = set(
    list(
        "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすず"
        "せぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴ"
        "ふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわ"
        "をんーゎゐゑゕゖゔ"
    )
)

KANJI_RX = re.compile(r"[\u4e00-\u9fff]")


class CodeFrequency(TypedDict):
    katakana: int
    hiragana: int
    kanji: int
    other: int


def code_frequencies(text: str) -> tuple[CodeFrequency, dict[str, int]]:
    """
    Count Japanese character types and individual character frequencies.

    Analyzes a Japanese text string to count occurrences of different
    character types (katakana, hiragana, kanji, other) and tracks the
    frequency of each unique character.

    Args:
        text: Japanese text string to analyze.

    Returns:
        A tuple containing:
            - CodeFrequency: TypedDict with counts for each character type:
                - katakana: Count of katakana characters
                - hiragana: Count of hiragana characters
                - kanji: Count of kanji characters
                - other: Count of all other characters
            - dict[str, int]: Frequency count for each unique character

    Examples:
        >>> freq, chars = code_frequencies("こんにちは世界")
        >>> freq["hiragana"]
        5
        >>> freq["kanji"]
        2
        >>> chars["ん"]
        1

        >>> freq2, chars2 = code_frequencies("カタカナABC")
        >>> freq2["katakana"]
        4
        >>> freq2["other"]  # Latin letters
        3
    """
    unigrams = Counter(text)
    cmap: CodeFrequency = {"katakana": 0, "hiragana": 0, "kanji": 0, "other": 0}
    for char, cnt in unigrams.items():
        if char in KATAKANA:
            cmap["katakana"] += cnt
        elif char in HIRAGANA:
            cmap["hiragana"] += cnt
        elif KANJI_RX.match(char):
            cmap["kanji"] += cnt
        else:
            cmap["other"] += cnt
    return cmap, dict(unigrams)


def is_katakana_sentence(text: str) -> bool:
    """
    Determine if text is predominantly katakana without hiragana.

    Uses multiple heuristics to identify katakana-only sentences while
    filtering out false positives like repeated characters, onomatopoeia,
    or very short exclamations.

    Args:
        text: Japanese sentence to analyze.

    Returns:
        True if the sentence is predominantly katakana (>50% katakana ratio,
        no hiragana) and passes quality heuristics. False otherwise.

    Heuristics applied:
        - Rejects very short text (<8 chars)
        - Rejects short text ending with punctuation marks
        - Rejects text with low character diversity
        - Rejects text with excessive character/bigram repetition
        - Rejects text matching onomatopoeia patterns

    Examples:
        >>> is_katakana_sentence("カタカナダケノブンショウデス")
        True

        >>> is_katakana_sentence("カタカナとひらがな")
        False

        >>> is_katakana_sentence("ワンワン")  # Onomatopoeia
        False

        >>> is_katakana_sentence("ア！")  # Too short
        False
    """
    cmap, unigram_types = code_frequencies(text)
    bigram_counts = Counter("".join(p) for p in pairwise(text))

    total_chars = cmap["katakana"] + cmap["hiragana"] + cmap["kanji"] + cmap["other"]
    katakana_ratio = cmap["katakana"] / total_chars

    if cmap["hiragana"] == 0 and katakana_ratio > 0.5:
        if (
            len(text) < 8
            or (len(text) < 10 and re.search(r"ッ?.?[？！]」$", text[-3:]))
            or (len(text) < 100 and len(unigram_types) / len(text) < 0.5)
            or max(bigram_counts.values(), default=0) / len(text) > 0.5
            or len(re.findall(r"(.)\1+", text)) / len(text) > 0.1
            or len(re.findall(r"(..)ッ?\1", text)) / len(text) > 0.1
        ):
            return False
        return True
    else:
        return False


def split_list(lst: list[str], n: int):
    """
    Yield successive n-sized chunks from lst.

    Examples:
        >>> list(split_list(['a','b','c','d'], 2))
        [['a', 'b'], ['c', 'd']]
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_corpus_core(
    all_metadata: pl.DataFrame | None,
    language: str,
    chunksize: int,
    min_chunksize: int,
    chunks: int,
    tokenizer_type: str,
    dictionary_type: str,
    tokenizer_factory: Callable[..., TokenizerFactory],
) -> tuple[list[str], pl.DataFrame]:
    """
    Pure core for corpus creation without Streamlit. Builds docs and metadata.

    Args:
        all_metadata: Optional metadata table with at least filename/author/title fields.
        language: "Japanese" or "English".
        chunksize: Target tokens per chunk.
        min_chunksize: Minimum tokens to keep a chunk.
        chunks: Max chunks per document (0 = unlimited).
        tokenizer_type: Tokenizer engine identifier.
        dictionary_type: Dictionary/model identifier for the tokenizer.
        tokenizer_factory: Callable that returns an object with `.tokenizer` having `.tokenize(str) -> list[str]`.

    Returns:
        (docs, metadata) where docs are chunk strings and metadata is a Polars DataFrame.
    """
    from topic_modeling_streamlit.utils import validate_chunking_settings

    validate_chunking_settings(chunksize, min_chunksize)

    docs: list[str] = []
    labels: list[str] = []
    filenames: list[str] = []
    authors: list[str] = []

    if language == "Japanese":
        tokenizer_obj = tokenizer_factory(
            tokenizer_type=tokenizer_type,
            dictionary_type=dictionary_type,
            language=language,
        )
        tokenizer = tokenizer_obj.tokenizer
        sep = ""
        files = list(
            Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob("*.txt")
        )
    else:
        tokenizer_obj = tokenizer_factory(
            tokenizer_type=tokenizer_type,
            dictionary_type=dictionary_type,
            language=language,
            features=["text_with_ws"],
        )
        tokenizer = tokenizer_obj.tokenizer
        sep = " "
        files = list(Path("./standard-ebooks-selection/").glob("*.txt"))

    for file in files:
        title = file.stem
        with open(file, encoding="utf-8") as f:
            doc: list[str] = []
            tokens_chunk: list[str] = []
            running_count = 0
            for paragraph in f.read().splitlines():
                if chunks > 0 and running_count >= chunks:
                    break
                if not paragraph:
                    continue
                if language == "Japanese":
                    if is_katakana_sentence(paragraph):
                        paragraph = jaconv.kata2hira(paragraph)

                tokens = (
                    tokenizer.tokenize(paragraph)
                    if language == "Japanese"
                    else paragraph.split()
                )

                if len(tokens) >= chunksize:
                    if tokens_chunk:
                        doc.append(sep.join(tokens_chunk))
                        running_count += 1
                    xs = list(split_list(tokens, chunksize))
                    last_chunk = xs.pop()
                    doc.extend(sep.join(x) for x in xs)
                    running_count += len(xs)
                    tokens_chunk = last_chunk if len(last_chunk) < chunksize else []
                elif len(tokens) + len(tokens_chunk) > chunksize:
                    doc.append(sep.join(tokens_chunk))
                    running_count += 1
                    tokens_chunk = tokens
                else:
                    tokens_chunk.extend(tokens)

            if tokens_chunk:
                doc.append(sep.join(tokens_chunk))

            if chunks > 0:
                doc = doc[:chunks]

            docs.extend(doc)
            labels.extend([title] * len(doc))
            filenames.extend([file.name] * len(doc))
            if all_metadata is None:
                author = title.split("_")[0]
            else:
                # Try to get the author from metadata; fallback to filename if missing
                matches = (
                    all_metadata.filter(pl.col("filename") == file.name)
                    .select("author")
                    .head(1)
                    .to_series()
                    .to_list()
                )
                author = matches[0] if matches else title.split("_")[0]
            authors.extend([author] * len(doc))

    # min-size filtering
    if language == "Japanese":
        filtered = [
            (d, lab, fn, auth)
            for d, lab, fn, auth in zip(docs, labels, filenames, authors)
            if len(tokenizer.tokenize(d)) >= min_chunksize
        ]
    else:
        filtered = [
            (d, lab, fn, auth)
            for d, lab, fn, auth in zip(docs, labels, filenames, authors)
            if len(d.split(" ")) >= min_chunksize
        ]
    if filtered:
        docs, labels, filenames, authors = map(list, zip(*filtered))
    else:
        docs, labels, filenames, authors = [], [], [], []

    base = pl.DataFrame(
        {
            "author": authors,
            "label": labels,
            "filename": filenames,
            "length": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )

    if all_metadata is not None and not all_metadata.is_empty():
        if language == "Japanese":
            meta = all_metadata.select("filename", "title", "genre", "year")
        else:
            meta = all_metadata.select("filename", "title", "genre")
        metadata = base.join(meta, on="filename", how="left")
    else:
        metadata = base.with_columns(
            pl.col("label").alias("title"),
            pl.lit("all").alias("genre"),
        )

    # Fail early if corpus production yielded no chunks--surface a clear error.
    if not docs:
        raise ValueError(
            "No document chunks were produced. Check that text files exist in the expected directory "
            "and that tokenizer/chunking options (chunksize, min_chunksize, surface_filter) are valid."
        )

    return docs, metadata


def get_metadata(language: str = "Japanese") -> pl.DataFrame | None:
    """
    Load metadata for Aozora (Japanese) or Standard Ebooks (English).
    Delegates to topic_modeling_streamlit.utils.get_metadata_df.
    """
    from topic_modeling_streamlit.utils import get_metadata_df

    return get_metadata_df(language, ui_warn=False)


def chunk_tokens(
    file: Path,
    tagger,
    metadata: pl.DataFrame,
    lemma: bool = False,
    remove_proper_nouns: bool = False,
    chunk_size: int = 2000,
) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Split a text into ~chunk_size-token chunks using sentence boundaries.
    Returns chunk labels, authors, and token chunks.
    """
    with open(file, encoding="utf-8") as f:
        filename = file.stem
        text = f.read().replace("。", "。\n").replace("\n\n", "\n")
        sentences = text.splitlines()

    labels: list[str] = []
    authors: list[str] = []
    chunks: list[list[str]] = [[]]

    # author_ja is present in the Aozora groups.csv metadata
    author = (
        metadata.filter(pl.col("filename") == file.name)
        .select(pl.col("author_ja"))
        .head(1)
        .to_series()
        .to_list()[0]
    )

    current_chunk = 0
    for sentence in sentences:
        tokens = tokenize(
            sentence, tagger, lemma=lemma, remove_proper_nouns=remove_proper_nouns
        )
        chunks[current_chunk].extend(tokens)
        if len(chunks[current_chunk]) >= chunk_size:
            labels.append(f"{filename}_{current_chunk}")
            authors.append(author)
            current_chunk += 1
            chunks.append([])
    if chunks and len(chunks[-1]) < chunk_size:
        chunks.pop()
    return labels, authors, chunks


def create_chunked_data(
    all_metadata: pl.DataFrame | None,
    chunksize: int = 2000,
) -> tuple[pl.DataFrame, list[list[str]], list[list[str]]]:
    """
    Chunk Aozora Japanese texts into token lists for the gensim app.

    Returns:
        metadata: DataFrame with author, label, filename, length, docid, title, genre, year
        docs: token lists (lemmatized, proper nouns removed)
        original_docs: token lists (surface forms)
    """
    if all_metadata is None:
        raise FileNotFoundError("Aozora metadata not found. Provide groups.csv.")

    tagger = get_tagger()

    labels: list[str] = []
    filenames: list[str] = []
    authors: list[str] = []
    docs: list[list[str]] = []
    original_docs: list[list[str]] = []

    for file in Path("./Aozora-Bunko-Fiction-Selection-2022-05-30/Plain/").glob(
        "*.txt"
    ):
        chunk_labels, chunk_authors, chunks = chunk_tokens(
            file,
            tagger,
            all_metadata,
            lemma=True,
            remove_proper_nouns=True,
            chunk_size=chunksize,
        )
        labels.extend(chunk_labels)
        filenames.extend([file.name] * len(chunk_labels))
        authors.extend(chunk_authors)
        docs.extend(chunks)

        _, _, original_chunks = chunk_tokens(
            file, tagger, all_metadata, lemma=False, chunk_size=chunksize
        )
        original_docs.extend(original_chunks)

    base = pl.DataFrame(
        {
            "author": authors,
            "label": labels,
            "filename": filenames,
            "length": [len(d) for d in docs],
            "docid": range(len(docs)),
        }
    )
    enriched = all_metadata.select(
        pl.col("filename", "genre", "year"),
        pl.col("author_ja").alias("author"),
        pl.col("title_ja").alias("title"),
    ).join(base.select(pl.col("filename", "label", "docid", "length")), on="filename")
    return enriched, docs, original_docs
