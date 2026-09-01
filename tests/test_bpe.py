from collections import Counter

from bpe.tokenizer import (
    END_OF_WORD,
    BPETokenizer,
    apply_merge,
    count_pairs,
    merge_word,
    pretokenize,
)


# ---------- pretokenize ----------

def test_pretokenize_single_word():
    assert pretokenize("low") == [["l", "o", "w", END_OF_WORD]]


def test_pretokenize_multiple_words():
    assert pretokenize("low lower") == [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
    ]


def test_pretokenize_collapses_whitespace():
    assert pretokenize("  low\tlower\n\nnewest  ") == [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
        ["n", "e", "w", "e", "s", "t", END_OF_WORD],
    ]


def test_pretokenize_empty_string():
    assert pretokenize("") == []


# ---------- count_pairs ----------

def test_count_pairs_single_word():
    corpus = [["l", "o", "w", END_OF_WORD]]
    assert count_pairs(corpus) == Counter({
        ("l", "o"): 1,
        ("o", "w"): 1,
        ("w", END_OF_WORD): 1,
    })


def test_count_pairs_multiple_words_shared_prefix():
    corpus = [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
    ]
    assert count_pairs(corpus) == Counter({
        ("l", "o"): 2,
        ("o", "w"): 2,
        ("w", END_OF_WORD): 1,
        ("w", "e"): 1,
        ("e", "r"): 1,
        ("r", END_OF_WORD): 1,
    })


def test_count_pairs_duplicates_are_summed():
    corpus = [["l", "o", "w", END_OF_WORD]] * 3
    assert count_pairs(corpus) == Counter({
        ("l", "o"): 3,
        ("o", "w"): 3,
        ("w", END_OF_WORD): 3,
    })


def test_count_pairs_single_char_word_contributes_nothing():
    corpus = [["a"]]
    assert count_pairs(corpus) == Counter()


def test_count_pairs_empty_corpus():
    assert count_pairs([]) == Counter()


def test_count_pairs_end_of_word_marker_is_a_regular_token():
    corpus = [["a", END_OF_WORD], ["b", END_OF_WORD]]
    assert count_pairs(corpus) == Counter({
        ("a", END_OF_WORD): 1,
        ("b", END_OF_WORD): 1,
    })


# ---------- merge_word ----------

def test_merge_word_simple():
    assert merge_word(["l", "o", "w", END_OF_WORD], ("l", "o")) == [
        "lo", "w", END_OF_WORD,
    ]


def test_merge_word_pair_not_present_returns_unchanged():
    word = ["l", "o", "w", END_OF_WORD]
    assert merge_word(word, ("x", "y")) == word


def test_merge_word_multiple_non_overlapping_occurrences():
    assert merge_word(["a", "b", "c", "a", "b"], ("a", "b")) == [
        "ab", "c", "ab",
    ]


def test_merge_word_overlapping_matches_are_not_double_counted():
    assert merge_word(["a", "a", "a"], ("a", "a")) == ["aa", "a"]


def test_merge_word_four_a_produces_two_aa():
    assert merge_word(["a", "a", "a", "a"], ("a", "a")) == ["aa", "aa"]


def test_merge_word_pair_at_end():
    assert merge_word(["a", "b", "c", "d"], ("c", "d")) == ["a", "b", "cd"]


def test_merge_word_pair_with_end_of_word_marker():
    assert merge_word(["l", "o", "w", END_OF_WORD], ("w", END_OF_WORD)) == [
        "l", "o", "w</w>",
    ]


def test_merge_word_single_token():
    assert merge_word(["a"], ("a", "b")) == ["a"]


def test_merge_word_empty():
    assert merge_word([], ("a", "b")) == []


def test_merge_word_does_not_mutate_input():
    original = ["l", "o", "w"]
    merge_word(original, ("l", "o"))
    assert original == ["l", "o", "w"]


# ---------- apply_merge ----------

def test_apply_merge_across_corpus():
    corpus = [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
    ]
    assert apply_merge(corpus, ("l", "o")) == [
        ["lo", "w", END_OF_WORD],
        ["lo", "w", "e", "r", END_OF_WORD],
    ]


def test_apply_merge_only_affects_words_containing_pair():
    corpus = [
        ["l", "o", "w", END_OF_WORD],
        ["c", "a", "t", END_OF_WORD],
    ]
    assert apply_merge(corpus, ("l", "o")) == [
        ["lo", "w", END_OF_WORD],
        ["c", "a", "t", END_OF_WORD],
    ]


def test_apply_merge_empty_corpus():
    assert apply_merge([], ("a", "b")) == []


def test_apply_merge_does_not_mutate_input():
    corpus = [["a", "b", "c"]]
    apply_merge(corpus, ("a", "b"))
    assert corpus == [["a", "b", "c"]]


# ---------- BPETokenizer.train ----------

def test_bpe_init_starts_empty():
    tok = BPETokenizer()
    assert tok.vocab == {}
    assert tok.merges == []


def test_train_tiny_repeating_corpus():
    # Corpus "ab ab" — trace by hand:
    #   pairs: (a,b)=2, (b,</w>)=2. Alphabetical tie-break → (a,b) wins.
    #   after merge: [['ab','</w>'], ['ab','</w>']]
    #   pairs: (ab,</w>)=2 → merge.
    #   corpus now [['ab</w>'], ['ab</w>']] — no more pairs. Stop.
    tok = BPETokenizer()
    tok.train("ab ab", vocab_size=100)
    assert tok.merges == [("a", "b"), ("ab", END_OF_WORD)]
    # Initial alphabet is {'</w>', 'a', 'b'} — sorts as: '</w>' < 'a' < 'b'
    # ('<' has ASCII 60, 'a' has 97, 'b' has 98).
    assert tok.vocab == {
        END_OF_WORD: 0,
        "a": 1,
        "b": 2,
        "ab": 3,
        "ab" + END_OF_WORD: 4,
    }


def test_train_sennrich_first_four_merges():
    # Canonical Sennrich et al. 2016 example, first four merges verified
    # against the paper's trace.
    text = (
        "low " * 5
        + "lower " * 2
        + "newest " * 6
        + "widest " * 3
    ).strip()
    tok = BPETokenizer()
    tok.train(text, vocab_size=100)
    assert tok.merges[:4] == [
        ("e", "s"),           # (e,s) count 9
        ("es", "t"),          # (es,t) count 9
        ("est", END_OF_WORD), # (est,</w>) count 9
        ("l", "o"),           # (l,o) count 7 — first alpha winner at count 7
    ]


def test_train_no_repeating_pair_yields_no_merges():
    # Single word "ab" appears once → (a,b) count 1, (b,</w>) count 1.
    # max_count is 1 → stop with no merges.
    tok = BPETokenizer()
    tok.train("ab", vocab_size=100)
    assert tok.merges == []
    # Vocab is just the initial alphabet.
    assert set(tok.vocab.keys()) == {"a", "b", END_OF_WORD}


def test_train_respects_vocab_size_upper_bound():
    text = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3).strip()
    tok = BPETokenizer()
    tok.train(text, vocab_size=12)
    assert len(tok.vocab) <= 12


def test_train_is_deterministic():
    text = "low lower newest widest " * 3
    a, b = BPETokenizer(), BPETokenizer()
    a.train(text, vocab_size=50)
    b.train(text, vocab_size=50)
    assert a.vocab == b.vocab
    assert a.merges == b.merges


def test_train_empty_text_leaves_empty_tokenizer():
    tok = BPETokenizer()
    tok.train("", vocab_size=100)
    assert tok.vocab == {}
    assert tok.merges == []


def test_train_ids_assigned_in_learning_order():
    # After training, the very first learned merge should have vocab ID equal
    # to the size of the initial alphabet (i.e., appended right after it).
    text = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3).strip()
    tok = BPETokenizer()
    tok.train(text, vocab_size=100)
    initial_chars = {"l", "o", "w", "e", "r", "n", "s", "t", "i", "d", END_OF_WORD}
    first_merge = tok.merges[0]
    first_merged_token = first_merge[0] + first_merge[1]
    assert tok.vocab[first_merged_token] == len(initial_chars)


def test_train_overwrites_prior_training():
    tok = BPETokenizer()
    tok.train("ab ab ab", vocab_size=100)
    first_run_merges = tok.merges.copy()
    first_run_vocab = tok.vocab.copy()

    tok.train("cd cd cd", vocab_size=100)
    assert tok.merges != first_run_merges
    assert tok.vocab != first_run_vocab
    assert "a" not in tok.vocab  # 'a' is gone entirely — was never in second corpus


# ---------- BPETokenizer.tokenize ----------

def _make_trained_tokenizer():
    """Train on "low low lower" with vocab_size=8 — the dry-run example."""
    tok = BPETokenizer()
    tok.train("low low lower", vocab_size=8)
    return tok


def test_tokenize_known_word():
    tok = _make_trained_tokenizer()
    assert tok.tokenize("low") == ["low", END_OF_WORD]


def test_tokenize_partial_merge():
    tok = _make_trained_tokenizer()
    # "lower" → merges (l,o) and (lo,w), but 'e' and 'r' stay separate.
    assert tok.tokenize("lower") == ["low", "e", "r", END_OF_WORD]


def test_tokenize_multi_word():
    tok = _make_trained_tokenizer()
    assert tok.tokenize("lower low") == [
        "low", "e", "r", END_OF_WORD,
        "low", END_OF_WORD,
    ]


def test_tokenize_min_rank_beats_leftmost():
    # Manually construct a tokenizer where (a,b) has rank 0 and (x,y)
    # has rank 1. For word "xyab", leftmost pair is (x,y) but min-rank
    # picks (a,b) first.
    tok = BPETokenizer()
    tok.vocab = {
        END_OF_WORD: 0, "a": 1, "b": 2, "x": 3, "y": 4,
        "ab": 5, "xy": 6,
    }
    tok.merges = [("a", "b"), ("x", "y")]
    # Word "xyab" → ['x','y','a','b','</w>']
    # Pairs: (x,y)=rank1, (a,b)=rank0. Min-rank picks (a,b).
    # → ['x','y','ab','</w>'], then (x,y)=rank1 → ['xy','ab','</w>'].
    tokens = tok.tokenize("xyab")
    assert tokens == ["xy", "ab", END_OF_WORD]


def test_tokenize_no_merges_returns_characters():
    tok = BPETokenizer()
    tok.train("ab", vocab_size=100)  # no repeating pair → no merges
    assert tok.tokenize("ab") == ["a", "b", END_OF_WORD]


def test_tokenize_empty_string():
    tok = _make_trained_tokenizer()
    assert tok.tokenize("") == []


def test_tokenize_unknown_char_raises_keyerror():
    tok = _make_trained_tokenizer()
    # 'x' was never seen during training.
    tokens = tok.tokenize("lox")
    assert "x" in tokens
    import pytest
    with pytest.raises(KeyError):
        tok.encode("lox")


# ---------- BPETokenizer.encode ----------

def test_encode_known_word():
    tok = _make_trained_tokenizer()
    # vocab: {'</w>':0, 'e':1, 'l':2, 'o':3, 'r':4, 'w':5, 'lo':6, 'low':7}
    assert tok.encode("low") == [7, 0]


def test_encode_partial_merge():
    tok = _make_trained_tokenizer()
    assert tok.encode("lower") == [7, 1, 4, 0]


def test_encode_multi_word():
    tok = _make_trained_tokenizer()
    assert tok.encode("lower low") == [7, 1, 4, 0, 7, 0]


def test_encode_roundtrip_with_sennrich_corpus():
    text = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3).strip()
    tok = BPETokenizer()
    tok.train(text, vocab_size=100)
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0
    # Every ID in the output should be a valid vocab value.
    valid_ids = set(tok.vocab.values())
    assert all(i in valid_ids for i in ids)


def test_encode_is_deterministic():
    text = "low lower newest widest " * 3
    tok = BPETokenizer()
    tok.train(text, vocab_size=50)
    assert tok.encode("lower newest") == tok.encode("lower newest")
