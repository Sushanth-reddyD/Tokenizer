from bpe.tokenizer import END_OF_WORD, pretokenize


def test_pretokenize_single_word():
    assert pretokenize("low") == [["l", "o", "w", END_OF_WORD]]


def test_pretokenize_multiple_words():
    assert pretokenize("low lower") == [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
    ]


def test_pretokenize_collapses_whitespace():
    # Runs of spaces, tabs, and newlines all collapse.
    assert pretokenize("  low\tlower\n\nnewest  ") == [
        ["l", "o", "w", END_OF_WORD],
        ["l", "o", "w", "e", "r", END_OF_WORD],
        ["n", "e", "w", "e", "s", "t", END_OF_WORD],
    ]


def test_pretokenize_empty_string():
    assert pretokenize("") == []
