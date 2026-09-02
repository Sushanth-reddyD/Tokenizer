from collections import Counter

import pytest

from wordpiece.pretokenizer import (
    CONTINUATION_PREFIX,
    _is_punctuation,
    _split_on_punctuation,
    _to_wordpiece_chars,
    pretokenize,
)
from wordpiece.tokenizer import (
    apply_merge,
    build_corpus,
    count_pairs,
    count_tokens,
    merge_word,
    score_pairs,
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


# ---------- build_corpus ----------

def test_build_corpus_basic():
    assert build_corpus("low low") == {("l", "##o", "##w"): 2}


def test_build_corpus_distinct_words():
    corpus = build_corpus("low high")
    assert corpus[("l", "##o", "##w")] == 1
    assert corpus[("h", "##i", "##g", "##h")] == 1


def test_build_corpus_empty():
    assert build_corpus("") == {}


def test_build_corpus_punctuation():
    """Punctuation splits into its own word."""
    corpus = build_corpus("hi!")
    assert ("h", "##i") in corpus
    assert ("!",) in corpus


# ---------- count_pairs ----------

def test_count_pairs_basic():
    corpus = {("l", "##o", "##w"): 2}
    assert count_pairs(corpus) == Counter({
        ("l", "##o"): 2,
        ("##o", "##w"): 2,
    })


def test_count_pairs_sums_across_words():
    corpus = {("l", "##o", "##w"): 1, ("l", "##o", "##n", "##g"): 1}
    pairs = count_pairs(corpus)
    assert pairs[("l", "##o")] == 2


def test_count_pairs_single_token_word():
    assert count_pairs({("!",): 5}) == Counter()


def test_count_pairs_empty():
    assert count_pairs({}) == Counter()


# ---------- count_tokens ----------

def test_count_tokens_basic():
    corpus = {("l", "##o", "##w"): 2}
    assert count_tokens(corpus) == Counter({"l": 2, "##o": 2, "##w": 2})


def test_count_tokens_sums_across_words():
    corpus = {("l", "##o", "##w"): 1, ("l", "##o", "##n", "##g"): 1}
    tokens = count_tokens(corpus)
    assert tokens["l"] == 2
    assert tokens["##o"] == 2
    assert tokens["##w"] == 1
    assert tokens["##n"] == 1


def test_count_tokens_empty():
    assert count_tokens({}) == Counter()


# ---------- score_pairs ----------

def test_score_pairs_always_together():
    """Tokens that always appear together score highest."""
    pair_counts = Counter({("##s", "##t"): 1})
    token_counts = Counter({"##s": 1, "##t": 1})
    scores = score_pairs(pair_counts, token_counts)
    assert scores[("##s", "##t")] == pytest.approx(1.0)


def test_score_pairs_independent_tokens():
    """Common tokens that co-occur proportionally score low."""
    pair_counts = Counter({("l", "##o"): 3})
    token_counts = Counter({"l": 3, "##o": 3})
    scores = score_pairs(pair_counts, token_counts)
    assert scores[("l", "##o")] == pytest.approx(1 / 3)


def test_score_pairs_low_lower_lowest():
    """Hand-verified: 'low lower lowest' — WordPiece picks (##s,##t) first."""
    corpus = build_corpus("low lower lowest")
    pair_counts = count_pairs(corpus)
    token_counts = count_tokens(corpus)
    scores = score_pairs(pair_counts, token_counts)
    best = max(scores, key=scores.get)
    assert best == ("##s", "##t")
    assert scores[best] == pytest.approx(1.0)


def test_score_pairs_bpe_would_disagree():
    """BPE picks (l,##o) (count=3), WordPiece picks (##s,##t) (score=1.0)."""
    corpus = build_corpus("low lower lowest")
    pair_counts = count_pairs(corpus)
    bpe_best = max(pair_counts, key=pair_counts.get)
    assert bpe_best in [("l", "##o"), ("##o", "##w")]  # tied at 3

    token_counts = count_tokens(corpus)
    scores = score_pairs(pair_counts, token_counts)
    wp_best = max(scores, key=scores.get)
    assert wp_best == ("##s", "##t")
    assert wp_best != bpe_best


# ---------- merge_word ----------

def test_merge_word_start_of_word():
    word = ["p", "##l", "##a", "##y"]
    assert merge_word(word, ("p", "##l")) == ["pl", "##a", "##y"]


def test_merge_word_continuation():
    word = ["p", "##l", "##a", "##y"]
    assert merge_word(word, ("##l", "##a")) == ["p", "##la", "##y"]


def test_merge_word_preserves_prefix():
    """First token's ## status carries through to the merged token."""
    assert merge_word(["##a", "##b"], ("##a", "##b")) == ["##ab"]
    assert merge_word(["a", "##b"], ("a", "##b")) == ["ab"]


def test_merge_word_no_overlap():
    word = ["##a", "##a", "##a"]
    assert merge_word(word, ("##a", "##a")) == ["##aa", "##a"]


def test_merge_word_multiple_occurrences():
    word = ["##a", "##b", "##a", "##b"]
    assert merge_word(word, ("##a", "##b")) == ["##ab", "##ab"]


def test_merge_word_no_match():
    word = ["a", "##b", "##c"]
    assert merge_word(word, ("##x", "##y")) == ["a", "##b", "##c"]


def test_merge_word_single_token():
    assert merge_word(["a"], ("a", "##b")) == ["a"]


def test_merge_word_empty():
    assert merge_word([], ("a", "##b")) == []


# ---------- apply_merge ----------

def test_apply_merge_basic():
    corpus = {("l", "##o", "##w"): 2}
    result = apply_merge(corpus, ("l", "##o"))
    assert result == {("lo", "##w"): 2}


def test_apply_merge_accumulates():
    corpus = {("a", "##b", "##c"): 3, ("ab", "##c"): 5}
    result = apply_merge(corpus, ("a", "##b"))
    assert result == {("ab", "##c"): 8}


def test_apply_merge_does_not_mutate():
    corpus = {("l", "##o", "##w"): 2}
    apply_merge(corpus, ("l", "##o"))
    assert corpus == {("l", "##o", "##w"): 2}


def test_apply_merge_empty():
    assert apply_merge({}, ("a", "##b")) == {}
