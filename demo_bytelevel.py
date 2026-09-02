"""Demo: train byte-level BPE on tinyshakespeare, then encode/decode/save/load.

Parallel to demo.py (piece #1). Key differences to watch for:
  - Lossless roundtrip (no whitespace normalization needed)
  - Merges shown as byte content, not string pairs
  - Special tokens demonstrated
  - Higher chars/token than piece #1 (regex pretok splits less aggressively)
"""

import time
from pathlib import Path

from bytelevel.bytemap import encode_token
from bytelevel.tokenizer import ByteLevelBPETokenizer

DATA_PATH = Path("data/input.txt")
MODEL_DIR = Path("data/bytelevel_model")
VOCAB_SIZE = 500
SPECIAL_TOKENS = ["<|endoftext|>"]
SAMPLE = "But, soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun."


def main() -> None:
    text = DATA_PATH.read_text()
    print(f"Corpus: {len(text):,} chars, {len(text.encode()):,} bytes\n")

    # --- Train ---
    tok = ByteLevelBPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    elapsed = time.time() - t0
    num_merges = len(tok.merges)
    print(f"Training: {elapsed:.1f}s → {len(tok.vocab)} vocab "
          f"({num_merges} merges + 256 base + {len(tok.special_tokens)} special)")

    print("\nFirst 10 merges (most frequent byte pairs):")
    for i, (a, b) in enumerate(tok.merges[:10]):
        content = tok.vocab[256 + i]
        display = encode_token(content)
        print(f"  {i:3d}  ({a:>3d},{b:>3d}) → {256+i:>3d}  {display!r:>12}  {content!r}")

    print("\nLast 10 merges (least frequent of those learned):")
    for i, (a, b) in enumerate(tok.merges[-10:], num_merges - 10):
        content = tok.vocab[256 + i]
        display = encode_token(content)
        print(f"  {i:3d}  ({a:>3d},{b:>3d}) → {256+i:>3d}  {display!r:>12}  {content!r}")

    # --- Encode sample ---
    print(f"\nSample text:\n  {SAMPLE!r}\n")
    ids = tok.encode(SAMPLE)
    tokens = tok.tokenize(SAMPLE)
    display_tokens = [encode_token(t) for t in tokens]
    print(f"Tokens ({len(tokens)}): {display_tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"IDs    ({len(ids)}): {ids[:15]}{'...' if len(ids) > 15 else ''}")

    chars_per_token = len(SAMPLE) / len(tokens)
    print(f"\nCompression: {len(SAMPLE)} chars → {len(tokens)} tokens "
          f"({chars_per_token:.2f} chars/token)")

    # --- Decode roundtrip (LOSSLESS — unlike piece #1) ---
    decoded = tok.decode(ids)
    assert decoded == SAMPLE, (
        f"Roundtrip failed!\n  Expected: {SAMPLE!r}\n  Got:      {decoded!r}"
    )
    print("Roundtrip: decode(encode(text)) == text  (lossless — no normalization)")

    # --- Full corpus encode ---
    t0 = time.time()
    full_ids = tok.encode(text)
    encode_time = time.time() - t0
    full_decoded = tok.decode(full_ids)
    assert full_decoded == text, "Full corpus roundtrip failed!"
    corpus_cpt = len(text) / len(full_ids)
    print(f"\nFull corpus: {len(text):,} chars → {len(full_ids):,} tokens "
          f"({corpus_cpt:.2f} chars/token, encoded in {encode_time:.1f}s)")

    # --- Special tokens ---
    sample_with_special = f"To be, or not to be,<|endoftext|>That is the question."
    ids_ordinary = tok.encode(sample_with_special)
    ids_special = tok.encode(sample_with_special, encode_special_tokens=True)
    eot_id = tok.special_tokens["<|endoftext|>"]
    print(f"\nSpecial tokens demo:")
    print(f"  Input: {sample_with_special!r}")
    print(f"  encode(ordinary):           {len(ids_ordinary)} tokens, "
          f"EOT id {eot_id} present: {eot_id in ids_ordinary}")
    print(f"  encode(special_tokens=True): {len(ids_special)} tokens, "
          f"EOT id {eot_id} present: {eot_id in ids_special}")
    assert tok.decode(ids_special) == sample_with_special

    # --- Save / Load ---
    tok.save(MODEL_DIR)
    loaded = ByteLevelBPETokenizer.load(MODEL_DIR)
    assert loaded.encode(SAMPLE) == ids
    assert loaded.special_tokens == tok.special_tokens
    print(f"\nSaved to {MODEL_DIR}/ and reloaded — encode + special tokens match")

    # --- File format peek ---
    print(f"\nFile format samples:")
    with open(MODEL_DIR / "merges.txt") as f:
        lines = f.readlines()
    print(f"  merges.txt: {len(lines)} lines")
    print(f"    first: {lines[0].rstrip()!r}")
    print(f"    last:  {lines[-1].rstrip()!r}")

    import json
    with open(MODEL_DIR / "vocab.json") as f:
        vocab = json.load(f)
    print(f"  vocab.json: {len(vocab)} entries")
    space_key = [k for k, v in vocab.items() if v == 32]
    print(f"    byte 32 (space) stored as: {space_key[0]!r} → {vocab[space_key[0]]}")


if __name__ == "__main__":
    main()
