import math

import pytest

from unigram.tokenizer import (
    build_seed_vocab,
    compute_loss,
    compute_piece_scores,
    pretokenize,
    viterbi_tokenize,
)


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


# ---------- compute_loss ----------

def test_compute_loss_single_chunk():
    """Loss = sum of negative log-probs of the Viterbi segmentation."""
    vocab = {"a": -1.0, "b": -2.0}
    # "ab" → ["a", "b"], loss = -(-1.0) + -(-2.0) = 3.0
    assert math.isclose(compute_loss(["ab"], vocab), 3.0)


def test_compute_loss_multi_chunk():
    vocab = {"a": -1.0, "b": -2.0}
    # Two chunks: "ab" (loss 3.0) + "a" (loss 1.0) = 4.0
    assert math.isclose(compute_loss(["ab", "a"], vocab), 4.0)


def test_compute_loss_empty_corpus():
    vocab = {"a": -1.0}
    assert compute_loss([], vocab) == 0.0


# ---------- compute_piece_scores ----------

def test_piece_scores_useful_piece_has_positive_delta():
    """Removing a useful piece should increase loss (positive delta)."""
    vocab = {"a": -1.0, "b": -1.0, "ab": -0.5}
    # With "ab": "ab" → ["ab"], loss = 0.5
    # Without "ab": "ab" → ["a","b"], loss = 2.0
    # delta = 2.0 - 0.5 = 1.5
    scores = compute_piece_scores(["ab"], vocab)
    assert "ab" in scores
    assert math.isclose(scores["ab"], 1.5)


def test_piece_scores_excludes_single_chars():
    """Single-character pieces must never appear in scores."""
    vocab = {"a": -1.0, "b": -1.0, "ab": -0.5}
    scores = compute_piece_scores(["ab"], vocab)
    assert "a" not in scores
    assert "b" not in scores


def test_piece_scores_useless_piece_has_zero_delta():
    """A piece not used by Viterbi should have delta ~0."""
    vocab = {"a": -0.5, "b": -0.5, "ab": -10.0}
    # Viterbi picks ["a","b"] (score -1) over ["ab"] (score -10).
    # Removing "ab" doesn't change the segmentation.
    scores = compute_piece_scores(["ab"], vocab)
    assert math.isclose(scores["ab"], 0.0, abs_tol=1e-12)
