"""Byte-level BPE tokenizer (cl100k_base pretokenization, GPT-2/tiktoken lineage).

Where piece #1 merged characters, this merges *byte ids*. The consequences:

  - The base vocabulary is exactly 256 and fixed, not derived from the corpus.
    Every possible input is therefore representable, so encode() can never
    raise the way piece #1's does on an unseen character.
  - Tokens are sequences of byte ids, so a merged token cannot name itself by
    string concatenation. New ids are allocated explicitly instead.

The merge algorithm itself is unchanged from piece #1: count adjacent pairs,
fuse the most frequent one, repeat. Only the element type moved from str to int.

Built from scratch for learning.
"""

import json
import re
from collections import Counter
from pathlib import Path

from .bytemap import decode_token, encode_token
from .pretokenizer import pretokenize

Vocab = dict[int, bytes]

# A corpus is unique-chunk -> how many times that chunk occurred. Piece #1
# added this grouping later as an optimization (54s -> 12s on tinyshakespeare);
# here it is the native representation from the start, which is why there is
# one count_pairs instead of a public list version plus a private grouped twin.
Corpus = dict[tuple[int, ...], int]


def build_corpus(text: str) -> Corpus:
    """Pretokenize `text` into a frequency-grouped corpus of byte-id chunks.

    Iterating a bytes object already yields ints, so the conversion from
    pretokenize()'s list[bytes] to tuples of byte ids is direct.

    Example:
        build_corpus("the the cat")
        -> {(116, 104, 101): 1,        # 'the'
            (32, 116, 104, 101): 1,    # ' the'
            (32, 99, 97, 116): 1}      # ' cat'
    """
    corpus: Counter[tuple[int, ...]] = Counter()
    for chunk in pretokenize(text):
        corpus[tuple(chunk)] += 1
    return dict(corpus)


def count_pairs(corpus: Corpus) -> Counter[tuple[int, int]]:
    """Count adjacent id pairs across the corpus, weighted by chunk frequency.

    A chunk occurring 500 times contributes each of its pairs 500 times, which
    is what makes the grouped representation equivalent to walking every chunk
    occurrence individually — but proportional to the number of *unique* chunks.

    Chunks of length 1 contribute nothing: there is no adjacent pair.
    """
    pair_counts: Counter[tuple[int, int]] = Counter()
    for chunk, freq in corpus.items():
        for i in range(len(chunk) - 1):
            pair_counts[(chunk[i], chunk[i + 1])] += freq
    return pair_counts


def merge_chunk(
    chunk: tuple[int, ...], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of `pair` in `chunk` with `new_id`.

    Scans left to right and skips forward two positions after a hit, so
    overlapping matches are not double-counted:

        merge_chunk((97, 97, 97), (97, 97), 256) -> (256, 97)

    `new_id` has to be supplied. Piece #1 could derive the merged token by
    string concatenation ("e" + "s" -> "es"), but ids carry no such structure,
    so allocation is the caller's job — see BPETokenizer.train().
    """
    a, b = pair
    result: list[int] = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(chunk[i])
            i += 1
    return tuple(result)


def apply_merge(corpus: Corpus, pair: tuple[int, int], new_id: int) -> Corpus:
    """Return a new corpus with every occurrence of `pair` merged to `new_id`.

    Does not mutate the input.

    Frequencies are accumulated rather than assigned, because two distinct
    chunks can merge into the same tuple: with (1,2,3) and (256,3) both present,
    merging (1,2) -> 256 collapses them. Plain assignment would silently drop
    one chunk's count.
    """
    new_corpus: Corpus = {}
    for chunk, freq in corpus.items():
        merged = merge_chunk(chunk, pair, new_id)
        new_corpus[merged] = new_corpus.get(merged, 0) + freq
    return new_corpus


class ByteLevelBPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer.

    Attributes:
        vocab:  Mapping from token ID to the bytes that token spells. IDs 0-255
                are the raw byte values; learned merges are numbered from 256 up.
        merges: Ordered list of (a, b) ID pairs learned by train(). Position in
                the list is the merge's "rank" — earlier = learned first.

    Note the split of concerns between the two. `merges` holds the *structural*
    fact that IDs a and b fuse; `vocab` holds the *payload*, the bytes that ID
    spells. Piece #1 fused both into one string. Keeping them apart is what lets
    encode() work purely off `merges` and decode() work purely off `vocab`.

    A freshly constructed tokenizer is already the identity byte tokenizer: the
    256 base entries are present, so it encodes and decodes any text losslessly
    with zero compression. Piece #1's fresh instance had an empty vocab and
    raised on everything.
    """

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[int, int]] = []
        self.special_tokens: dict[str, int] = {}

    def add_special_tokens(self, tokens: list[str]) -> dict[str, int]:
        """Register special tokens with IDs beyond the BPE merge range.

        Each token gets the next available ID. Duplicates are silently skipped.
        Returns the full special_tokens mapping (including any previously added).
        """
        for s in tokens:
            if s not in self.special_tokens:
                sid = len(self.vocab)
                self.vocab[sid] = s.encode("utf-8")
                self.special_tokens[s] = sid
        return dict(self.special_tokens)

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Learn BPE merges from `text` until the vocab reaches `vocab_size`.

        Overwrites any prior training on this instance. Stops early when no pair
        repeats, since further merges would compress nothing — so the final vocab
        can be smaller than requested (same behaviour as piece #1).

        Unlike piece #1 there is no alphabet discovery pass: the base vocabulary
        is the 256 byte values, identical for every corpus, and it needs no
        sorting because the IDs *are* the byte values. Determinism is therefore
        structural, and the only tie-break left is between equally frequent
        pairs, resolved by taking the lexicographically smallest.

        Special tokens (if given) are added after BPE training with IDs beyond
        the merge range. They do not count toward vocab_size.

        Raises ValueError if `vocab_size` is below 256, which cannot represent
        the base bytes.
        """
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256 to hold the base bytes, "
                f"got {vocab_size}"
            )

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.special_tokens = {}

        corpus = build_corpus(text)
        num_merges = vocab_size - 256

        for _ in range(num_merges):
            pair_counts = count_pairs(corpus)
            if not pair_counts:
                break

            max_count = max(pair_counts.values())
            if max_count < 2:
                break

            best_pair = min(
                pair for pair, count in pair_counts.items() if count == max_count
            )

            new_id = len(self.vocab)
            corpus = apply_merge(corpus, best_pair, new_id)
            # The concatenation piece #1 did on token strings happens here
            # instead, on the bytes payload rather than on the ID.
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)

            if verbose:
                print(
                    f"merge {len(self.merges)}/{num_merges}: "
                    f"{best_pair} -> {new_id} "
                    f"({max_count} occurrences) {self.vocab[new_id]!r}"
                )

        if special_tokens:
            self.add_special_tokens(special_tokens)

    def encode(
        self, text: str, encode_special_tokens: bool = False
    ) -> list[int]:
        """Encode text into a list of integer token IDs.

        When encode_special_tokens is False (the default), the entire text is
        treated as ordinary — special token strings like '<|endoftext|>' get
        pretokenized and BPE-encoded as regular bytes. This is the safe default:
        user-supplied text should not accidentally become control tokens.

        When True, registered special tokens are recognised as atomic units,
        and the text between them is BPE-encoded separately.
        """
        if encode_special_tokens and self.special_tokens:
            return self._encode_with_specials(text)
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        """BPE-encode text without any special-token handling."""
        ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        ids: list[int] = []
        for chunk_bytes in pretokenize(text):
            chunk = tuple(chunk_bytes)
            while len(chunk) > 1:
                best_pair = None
                best_rank = len(self.merges)
                for i in range(len(chunk) - 1):
                    pair = (chunk[i], chunk[i + 1])
                    rank = ranks.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                chunk = merge_chunk(chunk, best_pair, 256 + best_rank)
            ids.extend(chunk)
        return ids

    def _encode_with_specials(self, text: str) -> list[int]:
        """Split text on special token boundaries, BPE-encode each gap."""
        pattern = "(" + "|".join(
            re.escape(s)
            for s in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Concatenates each token's byte content and decodes the result as UTF-8.

        Piece #1 had to replace </w> with spaces and strip trailing whitespace.
        Here whitespace is in-band (byte 32 is a regular token), so decode is
        pure concatenation + UTF-8 decoding — no post-processing needed.

        Uses errors='replace' so that invalid byte sequences (possible when
        decoding model-generated IDs) produce U+FFFD rather than crashing.

        Raises KeyError if an ID is not in the vocab.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def tokenize(
        self, text: str, encode_special_tokens: bool = False
    ) -> list[bytes]:
        """Segment text into BPE tokens (byte strings).

        Convenience wrapper around encode() — maps each ID back to its byte
        content via the vocab. Useful for inspecting what the tokenizer
        actually produces.

        Piece #1 had this relationship inverted: tokenize() did the work and
        returned strings, encode() wrapped it. Here encode() is primary
        because the merge algorithm works with integer IDs natively.
        """
        return [self.vocab[i] for i in self.encode(text, encode_special_tokens)]

    def save(self, directory: str | Path) -> None:
        """Persist the tokenizer to a directory.

        Writes three files following GPT-2 conventions:
          vocab.json          — {bytemap-encoded token string: ID} for every token
          merges.txt          — one merge per line, space-separated bytemap pair
          special_tokens.json — {string: ID}, only written when special tokens exist

        The bytemap (segment 2) is applied here: each token's raw bytes are
        converted to a printable, whitespace-free string via encode_token().
        This is the ONLY place the bytemap touches the data — train(), encode(),
        and decode() never see it.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        encoded_vocab = {
            encode_token(token_bytes): token_id
            for token_id, token_bytes in self.vocab.items()
        }
        with open(path / "vocab.json", "w") as f:
            json.dump(encoded_vocab, f, ensure_ascii=False, indent=2)

        with open(path / "merges.txt", "w") as f:
            for a, b in self.merges:
                f.write(
                    f"{encode_token(self.vocab[a])} {encode_token(self.vocab[b])}\n"
                )

        if self.special_tokens:
            with open(path / "special_tokens.json", "w") as f:
                json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "ByteLevelBPETokenizer":
        """Load a tokenizer from a directory written by save().

        Reconstructs vocab, merges, and special_tokens. The bytemap is inverted
        here: decode_token() converts each printable string back to raw bytes.
        """
        path = Path(directory)
        tok = cls()

        with open(path / "vocab.json") as f:
            raw_vocab: dict[str, int] = json.load(f)

        tok.vocab = {}
        bytes_to_id: dict[bytes, int] = {}
        for token_str, token_id in raw_vocab.items():
            token_bytes = decode_token(token_str)
            tok.vocab[token_id] = token_bytes
            bytes_to_id[token_bytes] = token_id

        tok.merges = []
        with open(path / "merges.txt") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a_str, b_str = line.split(" ")
                tok.merges.append((
                    bytes_to_id[decode_token(a_str)],
                    bytes_to_id[decode_token(b_str)],
                ))

        tok.special_tokens = {}
        special_path = path / "special_tokens.json"
        if special_path.exists():
            with open(special_path) as f:
                tok.special_tokens = json.load(f)

        return tok
