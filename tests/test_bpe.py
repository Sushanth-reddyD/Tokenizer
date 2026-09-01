from collections import Counter

from bpe.tokenizer import (
    END_OF_WORD,
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
    # (a, b) appears twice, non-overlapping
    assert merge_word(["a", "b", "c", "a", "b"], ("a", "b")) == [
        "ab", "c", "ab",
    ]


def test_merge_word_overlapping_matches_are_not_double_counted():
    # ['a','a','a'] merging (a,a) -> non-overlapping left-to-right → ['aa','a']
    assert merge_word(["a", "a", "a"], ("a", "a")) == ["aa", "a"]


def test_merge_word_four_a_produces_two_aa():
    # ['a','a','a','a'] merging (a,a) → ['aa','aa']
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
        ["c", "a", "t", END_OF_WORD],  # unchanged
    ]


def test_apply_merge_empty_corpus():
    assert apply_merge([], ("a", "b")) == []


def test_apply_merge_does_not_mutate_input():
    corpus = [["a", "b", "c"]]
    apply_merge(corpus, ("a", "b"))
    assert corpus == [["a", "b", "c"]]
