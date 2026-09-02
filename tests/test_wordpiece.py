from wordpiece.pretokenizer import (
    CONTINUATION_PREFIX,
    _is_punctuation,
    _split_on_punctuation,
    _to_wordpiece_chars,
    pretokenize,
)


# ---------- _is_punctuation ----------

def test_ascii_punctuation():
    for char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        assert _is_punctuation(char), f"{char!r} should be punctuation"


def test_letters_and_digits_are_not_punctuation():
    for char in "abcABC019":
        assert not _is_punctuation(char)


def test_unicode_punctuation():
    assert _is_punctuation("、")  # ideographic comma
    assert _is_punctuation("—")  # em dash


def test_space_is_not_punctuation():
    assert not _is_punctuation(" ")
    assert not _is_punctuation("\t")


# ---------- _split_on_punctuation ----------

def test_split_punct_basic():
    assert _split_on_punctuation("don't") == ["don", "'", "t"]


def test_split_punct_trailing():
    assert _split_on_punctuation("hello!") == ["hello", "!"]


def test_split_punct_leading():
    assert _split_on_punctuation("(hello") == ["(", "hello"]


def test_split_punct_multiple():
    assert _split_on_punctuation("a.b.c") == ["a", ".", "b", ".", "c"]


def test_split_punct_only():
    assert _split_on_punctuation("...") == [".", ".", "."]


def test_split_punct_no_punct():
    assert _split_on_punctuation("hello") == ["hello"]


def test_split_punct_empty():
    assert _split_on_punctuation("") == []


# ---------- _to_wordpiece_chars ----------

def test_chars_basic():
    assert _to_wordpiece_chars("play") == ["p", "##l", "##a", "##y"]


def test_chars_single():
    assert _to_wordpiece_chars("!") == ["!"]


def test_chars_empty():
    assert _to_wordpiece_chars("") == []


def test_chars_two():
    assert _to_wordpiece_chars("ab") == ["a", "##b"]


# ---------- pretokenize ----------

def test_pretokenize_basic():
    assert pretokenize("Hello world") == [
        ["H", "##e", "##l", "##l", "##o"],
        ["w", "##o", "##r", "##l", "##d"],
    ]


def test_pretokenize_punctuation_splits():
    assert pretokenize("Hello, world!") == [
        ["H", "##e", "##l", "##l", "##o"],
        [","],
        ["w", "##o", "##r", "##l", "##d"],
        ["!"],
    ]


def test_pretokenize_contraction():
    assert pretokenize("don't") == [
        ["d", "##o", "##n"],
        ["'"],
        ["t"],
    ]


def test_pretokenize_empty():
    assert pretokenize("") == []


def test_pretokenize_whitespace_only():
    assert pretokenize("   ") == []


def test_pretokenize_multiple_whitespace():
    """Multiple spaces collapse (str.split() behaviour)."""
    assert pretokenize("a   b") == [["a"], ["b"]]


def test_pretokenize_single_char_words():
    assert pretokenize("I a") == [["I"], ["a"]]


def test_pretokenize_continuation_prefix_is_correct():
    """Every non-first character carries the ## prefix."""
    words = pretokenize("abc def")
    for word in words:
        assert not word[0].startswith(CONTINUATION_PREFIX)
        for char in word[1:]:
            assert char.startswith(CONTINUATION_PREFIX)
