from collections import Counter

import pytest

from wordpiece.pretokenizer import (
    CONTINUATION_PREFIX,
    _is_punctuation,
    _split_on_punctuation,
    _to_wordpiece_chars,
    pretokenize,
    split_words,
)
from wordpiece.tokenizer import (
    WordPieceTokenizer,
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


# ---------- WordPieceTokenizer.train() ----------

def test_train_initial_vocab():
    """Initial vocab is the sorted set of unique character tokens."""
    tok = WordPieceTokenizer()
    tok.train("ab ab", 2)
    assert "a" in tok.vocab
    assert "##b" in tok.vocab
    assert tok.vocab["##b"] < tok.vocab["a"]  # '#' (35) < 'a' (97)


def test_train_first_four_merges():
    """Hand-verified merge sequence on 'low lower lowest'.

    WordPiece builds from the rare end:
      1. (##s, ##t) → ##st     score=1.000
      2. (##e, ##r) → ##er     score=0.500
      3. (##e, ##st) → ##est   score=1.000
      4. (##o, ##w) → ##ow     score=0.333
    """
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 11)

    assert "##st" in tok.vocab
    assert "##er" in tok.vocab
    assert "##est" in tok.vocab
    assert "##ow" in tok.vocab

    assert tok.vocab["##st"] == 7
    assert tok.vocab["##er"] == 8
    assert tok.vocab["##est"] == 9
    assert tok.vocab["##ow"] == 10


def test_train_wordpiece_differs_from_bpe():
    """BPE would merge (l,##o) first (count=3); WordPiece merges (##s,##t)."""
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 8)  # only 1 merge
    first_merged = [t for t in tok.vocab if len(t) > 3 or (not t.startswith("##") and len(t) > 1)]
    assert "##st" in tok.vocab  # WordPiece's first merge
    # If BPE were used, "lo" would be the first merge instead


def test_train_respects_vocab_size():
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 9)  # 7 initial + 2 merges
    assert len(tok.vocab) == 9


def test_train_stops_when_no_pairs():
    """Single-char words have no pairs to merge."""
    tok = WordPieceTokenizer()
    tok.train("a b c", 100)
    assert len(tok.vocab) == 3  # just a, b, c — no ## tokens, no pairs


def test_train_overwrites_prior():
    tok = WordPieceTokenizer()
    tok.train("aaa aaa", 100)
    first_vocab = dict(tok.vocab)
    tok.train("bbb bbb", 100)
    assert tok.vocab != first_vocab


def test_train_empty_text():
    tok = WordPieceTokenizer()
    tok.train("", 100)
    assert tok.vocab == {}


def test_train_deterministic_tie_break():
    """When scores tie, the lexicographically smallest pair wins."""
    tok = WordPieceTokenizer()
    # "low lower lowest" round 2: (##e,##r) and (##e,##st) both score 0.5
    tok.train("low lower lowest", 9)  # 7 initial + 2 merges
    assert tok.vocab["##st"] == 7   # merge 1
    assert tok.vocab["##er"] == 8   # merge 2 (ties with ##e+##st, ##er < ##est)


def test_train_merged_token_inherits_prefix():
    """Merged tokens keep the first token's ## status."""
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 11)
    assert "##st" in tok.vocab    # ## + ## → ##
    assert "##er" in tok.vocab    # ## + ## → ##
    assert "##ow" in tok.vocab    # ## + ## → ##


def test_train_vocab_ids_are_contiguous():
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 11)
    ids = sorted(tok.vocab.values())
    assert ids == list(range(len(ids)))


# ---------- split_words ----------

def test_split_words_basic():
    assert split_words("Hello world") == ["Hello", "world"]


def test_split_words_punctuation():
    assert split_words("Hello, world!") == ["Hello", ",", "world", "!"]


def test_split_words_empty():
    assert split_words("") == []


# ---------- WordPieceTokenizer.tokenize() ----------

@pytest.fixture
def trained_wp():
    """Tokenizer trained on 'low lower lowest' with enough vocab for merges."""
    tok = WordPieceTokenizer()
    tok.train("low lower lowest", 15)
    return tok


def test_tokenize_known_word(trained_wp):
    """'low' should tokenize using learned merges, not individual chars."""
    tokens = trained_wp.tokenize("low")
    assert all(t in trained_wp.vocab for t in tokens)
    assert "".join(t.removeprefix("##") for t in tokens) == "low"


def test_tokenize_greedy_longest_match():
    """Greedy picks 'play' over 'p' when 'play' is in vocab."""
    tok = WordPieceTokenizer()
    tok.vocab = {"p": 0, "##l": 1, "##a": 2, "##y": 3, "play": 4,
                 "##i": 5, "##n": 6, "##g": 7, "##ing": 8}
    assert tok.tokenize("playing") == ["play", "##ing"]


def test_tokenize_falls_back_to_chars():
    """When no multi-char match exists, uses single characters."""
    tok = WordPieceTokenizer()
    tok.vocab = {"x": 0, "##y": 1, "##z": 2}
    assert tok.tokenize("xyz") == ["x", "##y", "##z"]


def test_tokenize_unk_for_unknown_char():
    """If a character isn't in vocab at all, entire word becomes [UNK]."""
    tok = WordPieceTokenizer()
    tok.vocab = {"a": 0, "##b": 1}
    assert tok.tokenize("abc") == ["[UNK]"]


def test_tokenize_unk_only_affects_that_word():
    """[UNK] replaces one word; other words encode normally."""
    tok = WordPieceTokenizer()
    tok.vocab = {"h": 0, "##i": 1, "[UNK]": 2}
    tokens = tok.tokenize("hi xyz")
    assert tokens == ["h", "##i", "[UNK]"]


def test_tokenize_punctuation_splits():
    """Punctuation becomes its own word, matched separately."""
    tok = WordPieceTokenizer()
    tok.vocab = {"h": 0, "##i": 1, "!": 2}
    assert tok.tokenize("hi!") == ["h", "##i", "!"]


def test_tokenize_multiple_words(trained_wp):
    tokens = trained_wp.tokenize("low lower")
    assert "[UNK]" not in tokens


def test_tokenize_empty():
    tok = WordPieceTokenizer()
    tok.vocab = {"a": 0}
    assert tok.tokenize("") == []


def test_tokenize_reconstruction(trained_wp):
    """Stripping ## and joining must reproduce the original words."""
    text = "low lower lowest"
    tokens = trained_wp.tokenize(text)
    words = split_words(text)
    reconstructed_words = []
    current = []
    for t in tokens:
        if not t.startswith("##") and current:
            reconstructed_words.append("".join(current))
            current = []
        current.append(t.removeprefix("##"))
    if current:
        reconstructed_words.append("".join(current))
    assert reconstructed_words == words


# ---------- WordPieceTokenizer.encode() ----------

def test_encode_returns_ids(trained_wp):
    ids = trained_wp.encode("low")
    assert all(isinstance(i, int) for i in ids)


def test_encode_ids_match_vocab(trained_wp):
    tokens = trained_wp.tokenize("low")
    ids = trained_wp.encode("low")
    assert ids == [trained_wp.vocab[t] for t in tokens]


def test_encode_unk_raises_without_unk_in_vocab():
    """Without [UNK] in vocab, encoding unknown text raises KeyError."""
    tok = WordPieceTokenizer()
    tok.vocab = {"a": 0, "##b": 1}
    with pytest.raises(KeyError):
        tok.encode("abc")
