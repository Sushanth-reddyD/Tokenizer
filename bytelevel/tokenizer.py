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

from collections import Counter

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

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn BPE merges from `text` until the vocab reaches `vocab_size`.

        Overwrites any prior training on this instance. Stops early when no pair
        repeats, since further merges would compress nothing — so the final vocab
        can be smaller than requested (same behaviour as piece #1).

        Unlike piece #1 there is no alphabet discovery pass: the base vocabulary
        is the 256 byte values, identical for every corpus, and it needs no
        sorting because the IDs *are* the byte values. Determinism is therefore
        structural, and the only tie-break left is between equally frequent
        pairs, resolved by taking the lexicographically smallest.

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
