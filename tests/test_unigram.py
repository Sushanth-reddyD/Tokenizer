import math

import pytest

from unigram.tokenizer import build_seed_vocab, pretokenize, viterbi_tokenize


# ---------- pretokenize ----------

def test_pretokenize_simple_sentence():
    assert pretokenize("Hello world") == ["Hello", " world"]


def test_pretokenize_lossless():
    text = "Hello, world!  How are you?\n"
    assert "".join(pretokenize(text)) == text


def test_pretokenize_contractions():
    chunks = pretokenize("don't won't I'm")
    assert "don" in chunks
    assert "'t" in chunks
    assert "'m" in chunks


def test_pretokenize_empty():
    assert pretokenize("") == []


# ---------- build_seed_vocab ----------

def test_seed_vocab_contains_all_chars():
    vocab = build_seed_vocab("abc")
    assert "a" in vocab
    assert "b" in vocab
    assert "c" in vocab


def test_seed_vocab_contains_substrings():
    vocab = build_seed_vocab("ab")
    assert "a" in vocab
    assert "b" in vocab
    assert "ab" in vocab


def test_seed_vocab_log_probabilities_sum():
    """Each log-prob should be log(count/total) — verify for a known case."""
    vocab = build_seed_vocab("aa")
    # chunks: ["aa"]. substrings: "a" (pos 0), "aa", "a" (pos 1).
    # counts: "a"=2, "aa"=1. total=3.
    assert math.isclose(vocab["a"], math.log(2 / 3), rel_tol=1e-9)
    assert math.isclose(vocab["aa"], math.log(1 / 3), rel_tol=1e-9)


def test_seed_vocab_respects_max_seed_size():
    vocab = build_seed_vocab("abcdef", max_seed_size=3)
    assert "abc" in vocab
    assert "abcd" not in vocab


def test_seed_vocab_all_values_negative():
    """Log-probabilities must be negative (count < total for any piece)."""
    vocab = build_seed_vocab("hello world")
    assert all(v < 0 for v in vocab.values())


# ---------- viterbi_tokenize ----------

def test_viterbi_single_chars():
    """With only single-char vocab, each character is its own token."""
    vocab = {"a": -1.0, "b": -1.0, "c": -1.0}
    assert viterbi_tokenize("abc", vocab) == ["a", "b", "c"]


def test_viterbi_prefers_longer_higher_prob():
    """Viterbi should pick the higher-probability segmentation."""
    # "ab" as one piece (score -1) beats "a"+"b" (score -2).
    vocab = {"a": -1.0, "b": -1.0, "ab": -1.0}
    assert viterbi_tokenize("ab", vocab) == ["ab"]


def test_viterbi_prefers_higher_total_score():
    """Even if a long piece exists, per-total-score the split might win."""
    # "ab" score -10 (terrible), "a"+-"b" score -0.5+-0.5 = -1 (better).
    vocab = {"a": -0.5, "b": -0.5, "ab": -10.0}
    assert viterbi_tokenize("ab", vocab) == ["a", "b"]


def test_viterbi_empty_string():
    vocab = {"a": -1.0}
    assert viterbi_tokenize("", vocab) == []


def test_viterbi_missing_char_raises():
    vocab = {"a": -1.0}
    with pytest.raises(ValueError, match="Cannot segment"):
        viterbi_tokenize("ab", vocab)
