from collections import defaultdict
import re

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


def code_frequencies(text):
    cmap = {
        "katakana": 0,
        "hiragana": 0,
        "kanji": 0,
        "other": 0,
    }

    unigram_types = defaultdict(int)

    for c in text:
        unigram_types[c] += 1
        if c in KATAKANA:
            cmap["katakana"] += 1
        elif c in HIRAGANA:
            cmap["hiragana"] += 1
        elif KANJI_RX.match(c):
            cmap["kanji"] += 1
        else:
            cmap["other"] += 1

    return cmap, unigram_types


def is_katakana_sentence(text: str):
    """
    Identify if sentence is made up of katakana (ie. no hiragana)
    characters. In such a case, it should be converted to hiragana,
    processed, and then the original orth should be substituted back
    in.

    Note: This identification is an imperfect heuristic and should
    be replaced.
    """

    cmap, unigram_types = code_frequencies(text)

    bigram_types = defaultdict(int)
    for bigram in map(lambda a, b: "{}{}".format(a, b), text, text[1:]):
        bigram_types[bigram] += 1

    katakana_ratio = cmap["katakana"] / sum(cmap.values())

    if cmap["hiragana"] == 0 and katakana_ratio > 0.5:
        # oov_count = sum(1 for token in tokens if token['oov'])
        # proper_noun_chars = sum(len(token['orth']) for token in tokens
        #                         if token['pos2'] == '固有名詞')

        # if oov_count == 0 and proper_noun_chars / len(text) > 0.3:
        #     return False

        # if oov_count / len(tokens) > 0.2 or \
        if (
            len(text) < 8
            or (len(text) < 10 and re.search(r"ッ?.?[？！]」$", text[-3:]))
            or (len(text) < 100 and len(unigram_types) / len(text) < 0.5)
            or max(bigram_types.values()) / len(text) > 0.5
            or len(re.findall(r"(.)\1+", text)) / len(text) > 0.1
            or len(re.findall(r"(..)ッ?\1", text)) / len(text) > 0.1
        ):
            return False
        return True
    else:
        return False
