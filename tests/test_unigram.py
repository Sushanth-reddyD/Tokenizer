import math

from unigram.tokenizer import build_seed_vocab, pretokenize


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
