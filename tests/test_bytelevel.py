from collections import Counter

import pytest

from bytelevel.bytemap import (
    bytes_to_unicode,
    decode_token,
    encode_token,
    unicode_to_bytes,
)
from bytelevel.pretokenizer import pretokenize, split_text
from bytelevel.tokenizer import (
    ByteLevelBPETokenizer,
    apply_merge,
    build_corpus,
    count_pairs,
    merge_chunk,
)


# ---------- split_text: the lossless invariant ----------

LOSSLESS_CASES = [
    "Hello world",
    "Hello, world!",
    "don't DON'T It'S",
    "1234567",
    "a\n\nb",
    "hello   world",
    "café 🎉 中文",
    "  leading",
    "trailing   ",
    "",
    "x = 1  # note\n",
    "\t\ttabbed\r\n",
]


def test_split_text_is_lossless():
    """Joining the chunks must reproduce the input exactly.

    This is the property piece #1 could not offer, and everything else about
    byte-level BPE's lossless roundtrip rests on it.
    """
    for text in LOSSLESS_CASES:
        assert "".join(split_text(text)) == text, f"lost data on {text!r}"


# ---------- split_text: chunk boundaries ----------

def test_leading_space_attaches_to_word():
    assert split_text("Hello world") == ["Hello", " world"]


def test_punctuation_splits_off_words():
    assert split_text("Hello, world!") == ["Hello", ",", " world", "!"]


def test_contractions_split_case_insensitively():
    assert split_text("don't DON'T It'S") == [
        "don", "'t", " DON", "'T", " It", "'S",
    ]


def test_digits_capped_at_three():
    assert split_text("1234567") == ["123", "456", "7"]


def test_number_does_not_absorb_leading_space():
    """Unlike words, the \\p{N} branch has no optional leading space."""
    assert split_text("x = 1") == ["x", " =", " ", "1"]


def test_newline_runs_stay_together():
    assert split_text("a\n\nb") == ["a", "\n\n", "b"]


def test_whitespace_run_splits_before_final_space():
    """Only the last space of a run attaches to the following word."""
    assert split_text("hello   world") == ["hello", "  ", " world"]


def test_trailing_whitespace_is_its_own_chunk():
    assert split_text("trailing   ") == ["trailing", "   "]


def test_unicode_letters_and_emoji():
    assert split_text("café 🎉 中文") == ["café", " 🎉", " 中文"]


def test_empty_string():
    assert split_text("") == []


# ---------- pretokenize: UTF-8 encoding ----------

def test_pretokenize_encodes_utf8():
    assert pretokenize("café") == [b"caf\xc3\xa9"]


def test_pretokenize_multibyte_chunk():
    """A single emoji is one chunk but four UTF-8 bytes."""
    chunks = pretokenize("🎉")
    assert chunks == [b"\xf0\x9f\x8e\x89"]
    assert len(chunks[0]) == 4


def test_pretokenize_is_lossless():
    for text in LOSSLESS_CASES:
        assert b"".join(pretokenize(text)) == text.encode("utf-8")


def test_pretokenize_never_raises_on_unseen_characters():
    """The whole point of byte-level: no input can be out-of-vocabulary."""
    exotic = "𝕳𝖊𝖑𝖑𝖔 ᚠᚢᚦ ｱｲｳ \x00\x7f"
    assert b"".join(pretokenize(exotic)) == exotic.encode("utf-8")


# ---------- bytemap: shape of the map ----------

def test_map_covers_all_256_bytes():
    table = bytes_to_unicode()
    assert len(table) == 256
    assert set(table) == set(range(256))


def test_map_is_bijective():
    table = bytes_to_unicode()
    assert len(set(table.values())) == 256
    inverse = unicode_to_bytes()
    assert all(inverse[char] == b for b, char in table.items())


def test_split_of_self_mapped_and_shifted():
    """188 bytes are already safe; the other 68 get shifted to U+0100+."""
    table = bytes_to_unicode()
    self_mapped = [b for b, char in table.items() if ord(char) == b]
    shifted = [b for b, char in table.items() if ord(char) != b]
    assert len(self_mapped) == 188
    assert len(shifted) == 68
    assert all(ord(table[b]) >= 0x100 for b in shifted)


# ---------- bytemap: the serialization safety guarantee ----------

def test_no_output_character_is_whitespace():
    """merges.txt is space-separated and line-based. A whitespace character
    anywhere in the map would make the file ambiguous or truncated."""
    assert not any(char.isspace() for char in bytes_to_unicode().values())


def test_every_output_character_is_printable():
    """Control characters would corrupt the file; soft hyphen (173) would
    render as nothing and silently produce an unreadable vocab."""
    assert all(char.isprintable() for char in bytes_to_unicode().values())


# ---------- bytemap: specific mappings ----------

def test_dangerous_bytes_are_shifted():
    table = bytes_to_unicode()
    assert table[0] == "Ā"    # U+0100, NUL
    assert table[9] == "ĉ"    # U+0109, tab
    assert table[10] == "Ċ"   # U+010A, newline
    assert table[32] == "Ġ"   # U+0120, space — the one you see everywhere
    assert table[127] == "ġ"  # U+0121, DEL
    assert table[160] == "ł"  # NBSP
    assert table[173] == "Ń"  # U+0143, soft hyphen — last of the 68


def test_printable_bytes_map_to_themselves():
    table = bytes_to_unicode()
    for char in "!~aZ0123abcÿ":
        assert table[ord(char)] == char


# ---------- bytemap: encode_token / decode_token ----------

def test_encode_token_leading_space():
    """The signature of every GPT-2-family vocab file."""
    assert encode_token(b" world") == "Ġworld"


def test_encode_token_blank_line():
    assert encode_token(b"\n\n") == "ĊĊ"


def test_encode_token_empty():
    assert encode_token(b"") == ""


def test_token_roundtrip_every_single_byte():
    for b in range(256):
        token = bytes([b])
        assert decode_token(encode_token(token)) == token


def test_token_roundtrip_multibyte_utf8():
    for text in LOSSLESS_CASES:
        token = text.encode("utf-8")
        assert decode_token(encode_token(token)) == token


def test_token_roundtrip_all_bytes_at_once():
    token = bytes(range(256))
    assert decode_token(encode_token(token)) == token


def test_encoded_token_never_contains_a_space():
    """Directly locks the merges.txt parsing guarantee for real tokens."""
    for text in LOSSLESS_CASES + ["🎉 中文", "a\tb\nc"]:
        encoded = encode_token(text.encode("utf-8"))
        assert " " not in encoded
        assert "\n" not in encoded


def test_decode_token_rejects_foreign_characters():
    """A character outside the 256-char map means the file was not written
    by encode_token()."""
    with pytest.raises(KeyError):
        decode_token("🎉")


# ---------- bytemap: callers cannot corrupt the tables ----------

def test_returned_maps_are_copies():
    table = bytes_to_unicode()
    table[32] = "CORRUPTED"
    assert bytes_to_unicode()[32] == "Ġ"

    inverse = unicode_to_bytes()
    del inverse["Ġ"]
    assert unicode_to_bytes()["Ġ"] == 32


# ---------- build_corpus ----------

def test_build_corpus_maps_text_to_byte_ids():
    assert build_corpus("the the cat") == {
        (116, 104, 101): 1,        # 'the'
        (32, 116, 104, 101): 1,    # ' the'
        (32, 99, 97, 116): 1,      # ' cat'
    }


def test_build_corpus_groups_identical_chunks():
    """Note ' the' (leading space) is a different chunk from 'the'. Only the
    two space-prefixed occurrences group together."""
    assert build_corpus("the the the") == {
        (116, 104, 101): 1,
        (32, 116, 104, 101): 2,
    }


def test_build_corpus_multibyte_character():
    assert build_corpus("é") == {(195, 169): 1}


def test_build_corpus_empty_text():
    assert build_corpus("") == {}


# ---------- count_pairs ----------

def test_count_pairs_basic():
    assert count_pairs({(1, 2, 3): 1}) == Counter({(1, 2): 1, (2, 3): 1})


def test_count_pairs_weights_by_frequency():
    """A chunk seen 500 times contributes each of its pairs 500 times."""
    assert count_pairs({(1, 2): 500}) == Counter({(1, 2): 500})


def test_count_pairs_sums_across_chunks():
    assert count_pairs({(1, 2, 3): 2, (9, 1, 2): 5}) == Counter(
        {(1, 2): 7, (2, 3): 2, (9, 1): 5}
    )


def test_count_pairs_single_id_chunk_has_no_pairs():
    assert count_pairs({(42,): 100}) == Counter()


def test_count_pairs_empty_corpus():
    assert count_pairs({}) == Counter()


def test_count_pairs_counts_repeated_pair_within_chunk():
    assert count_pairs({(1, 2, 1, 2): 1}) == Counter({(1, 2): 2, (2, 1): 1})


# ---------- merge_chunk ----------

def test_merge_chunk_basic():
    assert merge_chunk((1, 2, 3), (1, 2), 256) == (256, 3)


def test_merge_chunk_does_not_overlap():
    """The (97, 97, 97) case: skip two after a hit, so the third 97 is left
    alone rather than being merged into a second pair."""
    assert merge_chunk((97, 97, 97), (97, 97), 256) == (256, 97)


def test_merge_chunk_multiple_occurrences():
    assert merge_chunk((1, 2, 9, 1, 2), (1, 2), 256) == (256, 9, 256)


def test_merge_chunk_at_start_and_end():
    assert merge_chunk((1, 2, 9), (1, 2), 256) == (256, 9)
    assert merge_chunk((9, 1, 2), (1, 2), 256) == (9, 256)


def test_merge_chunk_no_match_returns_equal_chunk():
    assert merge_chunk((1, 2, 3), (7, 8), 256) == (1, 2, 3)


def test_merge_chunk_single_id_and_empty():
    assert merge_chunk((5,), (1, 2), 256) == (5,)
    assert merge_chunk((), (1, 2), 256) == ()


def test_merge_chunk_returns_a_tuple():
    """Tuples are the native type here so chunks stay hashable — usable as
    corpus keys and, later, as encode cache keys."""
    assert isinstance(merge_chunk((1, 2), (1, 2), 256), tuple)


def test_merge_chunk_new_id_is_used_verbatim():
    """Unlike piece #1, the merged token is not derived from the pair."""
    assert merge_chunk((1, 2), (1, 2), 9999) == (9999,)


# ---------- apply_merge ----------

def test_apply_merge_fans_out_across_corpus():
    corpus = {(1, 2, 3): 4, (9, 1, 2): 7}
    assert apply_merge(corpus, (1, 2), 256) == {(256, 3): 4, (9, 256): 7}


def test_apply_merge_preserves_untouched_chunks():
    corpus = {(1, 2): 3, (7, 8): 5}
    assert apply_merge(corpus, (1, 2), 256) == {(256,): 3, (7, 8): 5}


def test_apply_merge_accumulates_colliding_chunks():
    """Two distinct chunks can merge to the same tuple. Their frequencies must
    sum — plain assignment would silently drop one."""
    corpus = {(1, 2, 3): 5, (256, 3): 7}
    assert apply_merge(corpus, (1, 2), 256) == {(256, 3): 12}


def test_apply_merge_does_not_mutate_input():
    corpus = {(1, 2, 3): 4}
    apply_merge(corpus, (1, 2), 256)
    assert corpus == {(1, 2, 3): 4}


def test_apply_merge_empty_corpus():
    assert apply_merge({}, (1, 2), 256) == {}


# ---------- ByteLevelBPETokenizer.train() ----------

def test_train_base_vocab_always_present():
    """Even before any merges, all 256 byte values are in the vocab."""
    tok = ByteLevelBPETokenizer()
    tok.train("hello", 256)
    assert len(tok.vocab) == 256
    assert tok.merges == []
    for i in range(256):
        assert tok.vocab[i] == bytes([i])


def test_train_rejects_vocab_size_below_256():
    tok = ByteLevelBPETokenizer()
    with pytest.raises(ValueError, match="at least 256"):
        tok.train("hello", 100)


def test_train_first_three_merges():
    """Hand-verified merge sequence on 'the cat sat on the mat'.

    Pair counts at each step:
      round 1: (97,116)=3 wins — merges 'a','t' → 256  (b'at')
      round 2: (104,101)=2 and (116,104)=2 tie → (104,101) wins by min()
               → 257  (b'he')
      round 3: (116,257)=2 wins — merges 't',257 → 258  (b'the')
      round 4: all pairs have count 1 → stop
    """
    tok = ByteLevelBPETokenizer()
    tok.train("the cat sat on the mat", 300)

    assert tok.merges == [(97, 116), (104, 101), (116, 257)]
    assert tok.vocab[256] == b"at"
    assert tok.vocab[257] == b"he"
    assert tok.vocab[258] == b"the"
    assert len(tok.vocab) == 259


def test_train_vocab_tracks_byte_content():
    """Each merged token's bytes equals the concatenation of its constituents."""
    tok = ByteLevelBPETokenizer()
    tok.train("the cat sat on the mat", 300)
    for a, b in tok.merges:
        merge_id = 256 + tok.merges.index((a, b))
        assert tok.vocab[merge_id] == tok.vocab[a] + tok.vocab[b]


def test_train_stops_when_no_pair_repeats():
    """'abcd' has all unique pairs (each count=1) → no merges."""
    tok = ByteLevelBPETokenizer()
    tok.train("abcd", 300)
    assert tok.merges == []
    assert len(tok.vocab) == 256


def test_train_respects_vocab_size_cap():
    """With vocab_size=257, only one merge happens even if more are possible."""
    tok = ByteLevelBPETokenizer()
    tok.train("the cat sat on the mat", 257)
    assert len(tok.merges) == 1
    assert tok.merges[0] == (97, 116)
    assert len(tok.vocab) == 257


def test_train_overwrites_prior_training():
    tok = ByteLevelBPETokenizer()
    tok.train("aaa aaa aaa", 300)
    first_merges = list(tok.merges)
    assert len(first_merges) > 0

    tok.train("bbb bbb bbb", 300)
    assert tok.merges != first_merges


def test_train_deterministic_tie_break():
    """When two pairs have equal count, the lexicographically smaller tuple wins."""
    tok = ByteLevelBPETokenizer()
    tok.train("the cat sat on the mat", 258)
    assert tok.merges[1] == (104, 101)  # (104,101) < (116,104)


def test_train_empty_text():
    tok = ByteLevelBPETokenizer()
    tok.train("", 300)
    assert tok.merges == []
    assert len(tok.vocab) == 256


def test_train_multibyte_utf8():
    """Merges happen on byte ids, so a 2-byte UTF-8 char is two base tokens."""
    tok = ByteLevelBPETokenizer()
    tok.train("éé éé éé", 257)
    # 'é' = bytes 0xC3 0xA9 = (195, 169)
    # The leading-space variants ' éé' contribute (32,195) once and (195,169)
    # twice each, but 'éé' (no space) also contributes (195,169) twice.
    # So (195,169) appears most frequently → first merge.
    assert tok.merges[0] == (195, 169)
    assert tok.vocab[256] == b"\xc3\xa9"  # the full UTF-8 'é'


def test_fresh_tokenizer_has_base_vocab():
    """A freshly constructed tokenizer already has the 256-byte base vocab."""
    tok = ByteLevelBPETokenizer()
    assert len(tok.vocab) == 256
    assert tok.vocab[0] == b"\x00"
    assert tok.vocab[32] == b" "
    assert tok.vocab[255] == b"\xff"


# ---------- ByteLevelBPETokenizer.encode() ----------

@pytest.fixture
def trained_tok():
    """Tokenizer trained on 'the cat sat on the mat' — reused across tests."""
    tok = ByteLevelBPETokenizer()
    tok.train("the cat sat on the mat", 300)
    return tok


def test_encode_uses_learned_merges(trained_tok):
    """'the cat' should use the merges: 'at'→256, 'he'→257, 'the'→258."""
    ids = trained_tok.encode("the cat")
    # "the" → 258 (fully merged), " cat" → 32, 99, 256
    assert ids == [258, 32, 99, 256]


def test_encode_training_corpus_roundtrip(trained_tok):
    """Encoding the training text and mapping back to bytes reproduces it."""
    text = "the cat sat on the mat"
    ids = trained_tok.encode(text)
    reconstructed = b"".join(trained_tok.vocab[i] for i in ids)
    assert reconstructed == text.encode("utf-8")


def test_encode_unseen_ascii(trained_tok):
    """Characters not in training data encode to raw byte IDs — no error."""
    ids = trained_tok.encode("xyz")
    assert ids == [120, 121, 122]  # ord('x'), ord('y'), ord('z')


def test_encode_unseen_emoji(trained_tok):
    """Emoji decomposes into its 4 UTF-8 bytes — never raises."""
    ids = trained_tok.encode("🎉")
    assert ids == [0xF0, 0x9F, 0x8E, 0x89]


def test_encode_unseen_multibyte(trained_tok):
    """'é' (U+00E9) = 2 UTF-8 bytes: 0xC3 0xA9."""
    ids = trained_tok.encode("é")
    assert ids == [0xC3, 0xA9]


def test_encode_empty_text(trained_tok):
    assert trained_tok.encode("") == []


def test_encode_single_byte(trained_tok):
    """A single ASCII char that has no merge: returns its byte value."""
    ids = trained_tok.encode("x")
    assert ids == [120]


def test_encode_min_rank_picks_earliest_merge(trained_tok):
    """'heat' contains both (104,101)→rank 1 and (97,116)→rank 0.
    Min-rank picks (97,116) first even though (104,101) is leftmost."""
    ids = trained_tok.encode("heat")
    # round 1: (97,116) rank 0 wins → (104, 101, 256)
    # round 2: (104,101) rank 1 → (257, 256)
    # = b'he' + b'at'
    assert ids == [257, 256]


def test_encode_fresh_tokenizer_returns_raw_bytes():
    """Without training, encode returns one ID per byte."""
    tok = ByteLevelBPETokenizer()
    ids = tok.encode("abc")
    assert ids == [97, 98, 99]


def test_encode_fresh_tokenizer_handles_multibyte():
    """Even without merges, multibyte UTF-8 just returns raw byte IDs."""
    tok = ByteLevelBPETokenizer()
    ids = tok.encode("中")
    assert ids == list("中".encode("utf-8"))


# ---------- ByteLevelBPETokenizer.tokenize() ----------

def test_tokenize_returns_byte_content(trained_tok):
    tokens = trained_tok.tokenize("the cat")
    assert tokens == [b"the", b" ", b"c", b"at"]


def test_tokenize_concatenation_is_lossless(trained_tok):
    """Joining tokenize() output reproduces the original text as bytes."""
    text = "the cat sat on the mat"
    assert b"".join(trained_tok.tokenize(text)) == text.encode("utf-8")


def test_tokenize_emoji_returns_individual_bytes(trained_tok):
    tokens = trained_tok.tokenize("🎉")
    assert tokens == [bytes([b]) for b in "🎉".encode("utf-8")]


# ---------- ByteLevelBPETokenizer.decode() ----------

def test_decode_reverses_encode(trained_tok):
    text = "the cat sat on the mat"
    assert trained_tok.decode(trained_tok.encode(text)) == text


def test_decode_roundtrip_unseen_text(trained_tok):
    text = "hello world"
    assert trained_tok.decode(trained_tok.encode(text)) == text


def test_decode_roundtrip_unicode(trained_tok):
    for text in ["café", "中文测试", "🎉🚀", "a\tb\nc"]:
        assert trained_tok.decode(trained_tok.encode(text)) == text


def test_decode_roundtrip_whitespace(trained_tok):
    """Whitespace is in-band — spaces, tabs, newlines all roundtrip."""
    text = "hello   world\n\nnew\tline"
    assert trained_tok.decode(trained_tok.encode(text)) == text


def test_decode_empty_ids(trained_tok):
    assert trained_tok.decode([]) == ""


def test_decode_single_base_byte(trained_tok):
    assert trained_tok.decode([65]) == "A"


def test_decode_merged_token(trained_tok):
    assert trained_tok.decode([258]) == "the"


def test_decode_unknown_id_raises(trained_tok):
    with pytest.raises(KeyError):
        trained_tok.decode([99999])


def test_decode_invalid_utf8_uses_replacement_char():
    """Model-generated IDs can produce invalid UTF-8. errors='replace'
    substitutes U+FFFD instead of crashing."""
    tok = ByteLevelBPETokenizer()
    # 0xC3 is the start of a 2-byte sequence, but 0x41 ('A') is not
    # a valid continuation byte.
    assert tok.decode([0xC3, 0x41]) == "�A"


def test_decode_fresh_tokenizer_roundtrip():
    """Even without training, decode(encode(text)) is lossless."""
    tok = ByteLevelBPETokenizer()
    text = "hello 世界"
    assert tok.decode(tok.encode(text)) == text
