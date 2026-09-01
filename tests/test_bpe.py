from collections import Counter

from bpe.tokenizer import END_OF_WORD, count_pairs, pretokenize


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
    # "low</w>" has 3 adjacent pairs: (l,o), (o,w), (w,</w>)
    corpus = [["l", "o", "w", END_OF_WORD]]
    assert count_pairs(corpus) == Counter({
        ("l", "o"): 1,
        ("o", "w"): 1,
        ("w", END_OF_WORD): 1,
    })


def test_count_pairs_multiple_words_shared_prefix():
    # "low</w>" and "lower</w>" share (l,o) and (o,w)
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
    # Same word 3 times means each of its pairs is counted 3 times.
    corpus = [["l", "o", "w", END_OF_WORD]] * 3
    assert count_pairs(corpus) == Counter({
        ("l", "o"): 3,
        ("o", "w"): 3,
        ("w", END_OF_WORD): 3,
    })


def test_count_pairs_single_char_word_contributes_nothing():
    # "a</w>" has 1 pair: (a,</w>). But a length-1 corpus entry with no </w>
    # would contribute zero pairs.
    corpus = [["a"]]  # deliberately no </w> — length 1 → no pairs
    assert count_pairs(corpus) == Counter()


def test_count_pairs_empty_corpus():
    assert count_pairs([]) == Counter()


def test_count_pairs_end_of_word_marker_is_a_regular_token():
    # </w> participates in pairs just like any other token.
    corpus = [["a", END_OF_WORD], ["b", END_OF_WORD]]
    assert count_pairs(corpus) == Counter({
        ("a", END_OF_WORD): 1,
        ("b", END_OF_WORD): 1,
    })
