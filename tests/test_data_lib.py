import pytest

from topic_modeling_streamlit.data_lib import (
    code_frequencies,
    create_corpus_core,
    is_katakana_sentence,
)


def test_code_frequencies_basic():
    text = "こんにちは世界ABC"
    freq, chars = code_frequencies(text)
    # 5 hiragana, 2 kanji, 3 Latin/other
    assert freq["hiragana"] == 5
    assert freq["kanji"] == 2
    assert freq["other"] == 3
    # individual counts
    assert chars["こ"] == 1
    assert chars["A"] == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("カタカナダケノブンショウデス", True),
        ("ア！", False),  # too short
        ("ワンワン", False),  # onomatopoeia
        ("カタカナとひらがな", False),  # mixed script
        ("カタカナカナカナカナカナカナナ", False),  # low diversity
    ],
)
def test_is_katakana_sentence(text, expected):
    assert is_katakana_sentence(text) is expected


def test_english_corpus_uses_fallback_metadata_when_csv_is_missing(
    tmp_path, monkeypatch
):
    class Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return text.split()

    class TokenizerFactory:
        def __init__(self, **kwargs):
            self.tokenizer = Tokenizer()

    monkeypatch.chdir(tmp_path)
    corpus_dir = tmp_path / "standard-ebooks-selection"
    corpus_dir.mkdir()
    (corpus_dir / "jane-austen_test.txt").write_text(
        "A clean text corpus for English tokenization.", encoding="utf-8"
    )

    docs, metadata = create_corpus_core(
        all_metadata=None,
        language="English",
        chunksize=3,
        min_chunksize=1,
        chunks=0,
        tokenizer_type="spaCy",
        dictionary_type="en_core_web_sm",
        tokenizer_factory=TokenizerFactory,
    )

    assert docs
    assert metadata.get_column("author").unique().to_list() == ["jane-austen"]
    assert metadata.get_column("genre").unique().to_list() == ["all"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("one two three", ["one two three"]),
        ("one two three four five six", ["one two three", "four five six"]),
        ("one two three four", ["one two three", "four"]),
        ("zero\none two three", ["zero", "one two three"]),
    ],
)
def test_corpus_keeps_full_final_chunks(tmp_path, monkeypatch, text, expected):
    class Factory:
        def __init__(self, **kwargs):
            self.tokenizer = self

        def tokenize(self, text):
            return text.split()

    monkeypatch.chdir(tmp_path)
    corpus_dir = tmp_path / "standard-ebooks-selection"
    corpus_dir.mkdir()
    (corpus_dir / "author_book.txt").write_text(text, encoding="utf-8")

    docs, metadata = create_corpus_core(
        None, "English", 3, 1, 0, "spaCy", "en_core_web_sm", Factory
    )

    assert docs == expected
    assert metadata.height == len(expected)


def test_empty_text_is_not_a_katakana_sentence():
    assert not is_katakana_sentence("")
