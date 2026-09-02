"""Demo: train WordPiece on tinyshakespeare, then encode/decode/save/load.

Parallel to demo.py (piece #1) and demo_bytelevel.py (piece #2).
Key differences to watch for:
  - Score-based merges (likelihood ratio, not frequency)
  - Greedy longest-match encoding (not merge replay)
  - [UNK] for unseen characters
  - Lossy roundtrip (whitespace collapsed, punctuation spaced)
"""

import time
from pathlib import Path

from wordpiece.tokenizer import WordPieceTokenizer

DATA_PATH = Path("data/input.txt")
MODEL_DIR = Path("data/wordpiece_model")
VOCAB_SIZE = 500
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
SAMPLE = "But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun."


def main() -> None:
    text = DATA_PATH.read_text()
    print(f"Corpus: {len(text):,} chars, {len(text.split()):,} words\n")

    # --- Train ---
    tok = WordPieceTokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    elapsed = time.time() - t0
    initial_chars = len([t for t in tok.vocab if len(t) <= 3])
    num_merges = len(tok.vocab) - initial_chars - len(tok.special_tokens)
    print(f"Training: {elapsed:.1f}s → {len(tok.vocab)} vocab "
          f"({num_merges} merges + {initial_chars} chars + {len(tok.special_tokens)} special)")

    # --- Show some learned tokens ---
    merged_tokens = sorted(
        [(t, tid) for t, tid in tok.vocab.items()
         if t not in tok.special_tokens and (len(t) > 3 or (len(t) > 1 and not t.startswith("##")))],
        key=lambda x: x[1],
    )
    print(f"\nFirst 10 merged tokens (by ID order):")
    for token, tid in merged_tokens[:10]:
        print(f"  {tid:4d}  {token!r}")
    print(f"\nLast 10 merged tokens:")
    for token, tid in merged_tokens[-10:]:
        print(f"  {tid:4d}  {token!r}")

    # --- Encode sample ---
    print(f"\nSample text:\n  {SAMPLE!r}\n")
    tokens = tok.tokenize(SAMPLE)
    ids = tok.encode(SAMPLE)
    print(f"Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"IDs    ({len(ids)}): {ids[:15]}{'...' if len(ids) > 15 else ''}")

    chars_per_token = len(SAMPLE) / len(tokens)
    print(f"\nCompression: {len(SAMPLE)} chars → {len(tokens)} tokens "
          f"({chars_per_token:.2f} chars/token)")

    # --- Decode roundtrip (LOSSY — like piece #1) ---
    decoded = tok.decode(ids)
    normalized = " ".join(SAMPLE.split())
    from wordpiece.pretokenizer import split_words
    normalized_wp = " ".join(split_words(SAMPLE))
    print(f"\nRoundtrip (lossy — punctuation gets extra spaces):")
    print(f"  Original:  {SAMPLE!r}")
    print(f"  Decoded:   {decoded!r}")
    assert decoded == normalized_wp

    # --- Full corpus encode ---
    t0 = time.time()
    full_ids = tok.encode(text)
    encode_time = time.time() - t0
    corpus_cpt = len(text) / len(full_ids)
    print(f"\nFull corpus: {len(text):,} chars → {len(full_ids):,} tokens "
          f"({corpus_cpt:.2f} chars/token, encoded in {encode_time:.1f}s)")

    # --- Special tokens demo ---
    sample_with_special = "[CLS] To be or not to be [SEP] That is the question [SEP]"
    ids_special = tok.encode(sample_with_special, encode_special_tokens=True)
    cls_id = tok.special_tokens["[CLS]"]
    sep_id = tok.special_tokens["[SEP]"]
    print(f"\nSpecial tokens demo:")
    print(f"  Input: {sample_with_special!r}")
    print(f"  Tokens: {tok.tokenize(sample_with_special, encode_special_tokens=True)[:12]}...")
    print(f"  [CLS]={cls_id} at position 0: {ids_special[0] == cls_id}")
    print(f"  [SEP]={sep_id} count: {ids_special.count(sep_id)}")

    # --- [UNK] demo ---
    unk_text = "hello 🎉 world"
    unk_tokens = tok.tokenize(unk_text)
    unk_id = tok.special_tokens["[UNK]"]
    print(f"\n[UNK] demo:")
    print(f"  Input: {unk_text!r}")
    print(f"  Tokens: {unk_tokens}")
    print(f"  Emoji becomes [UNK]: {'[UNK]' in unk_tokens}")

    # --- Save / Load ---
    tok.save(MODEL_DIR)
    loaded = WordPieceTokenizer.load(MODEL_DIR)
    assert loaded.encode(SAMPLE) == ids
    assert loaded.special_tokens == tok.special_tokens
    print(f"\nSaved to {MODEL_DIR}/ and reloaded — encode + special tokens match")

    # --- File format peek ---
    with open(MODEL_DIR / "vocab.txt") as f:
        lines = f.readlines()
    print(f"\nFile format:")
    print(f"  vocab.txt: {len(lines)} lines (one token per line, line# = ID)")
    print(f"    line 0: {lines[0].rstrip()!r}")
    print(f"    line 1: {lines[1].rstrip()!r}")
    print(f"    last:   {lines[-1].rstrip()!r}")


if __name__ == "__main__":
    main()
