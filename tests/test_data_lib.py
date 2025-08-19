import pytest

from topic_modeling_streamlit.data_lib import code_frequencies, is_katakana_sentence


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
