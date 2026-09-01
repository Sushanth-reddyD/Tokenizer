"""Demo: train char-level BPE on tinyshakespeare, then encode/decode/save/load."""

import time
from pathlib import Path

from bpe.tokenizer import BPETokenizer

DATA_PATH = Path("data/input.txt")
MODEL_DIR = Path("data/bpe_model")
VOCAB_SIZE = 500
SAMPLE = "But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun."


def main() -> None:
    text = DATA_PATH.read_text()
    print(f"Corpus: {len(text):,} chars, {len(text.split()):,} words\n")

    # --- Train ---
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=VOCAB_SIZE)
    elapsed = time.time() - t0
    print(f"Training: {elapsed:.1f}s → {len(tok.vocab)} vocab, {len(tok.merges)} merges")

    print("\nFirst 10 merges (most frequent pairs):")
    for i, (a, b) in enumerate(tok.merges[:10]):
        print(f"  {i:3d}  {a!r:>10} + {b!r:<10} → {a + b!r}")

    print("\nLast 10 merges (least frequent of those learned):")
    for i, (a, b) in enumerate(tok.merges[-10:], len(tok.merges) - 10):
        print(f"  {i:3d}  {a!r:>10} + {b!r:<10} → {a + b!r}")

    # --- Encode ---
    print(f"\nSample text:\n  {SAMPLE!r}\n")
    tokens = tok.tokenize(SAMPLE)
    ids = tok.encode(SAMPLE)
    print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"IDs    ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")

    chars_per_token = len(SAMPLE) / len(tokens)
    print(f"\nCompression: {len(SAMPLE)} chars → {len(tokens)} tokens ({chars_per_token:.2f} chars/token)")

    # --- Decode roundtrip ---
    decoded = tok.decode(ids)
    assert decoded == SAMPLE, f"Roundtrip failed!\n  Expected: {SAMPLE!r}\n  Got:      {decoded!r}"
    print("Roundtrip: ✓ decode(encode(text)) == text")

    # --- Full corpus roundtrip ---
    # pretokenize normalizes all whitespace (newlines, tabs, runs) to single
    # spaces, so we compare against the normalized form.
    normalized = " ".join(text.split())
    t0 = time.time()
    full_ids = tok.encode(text)
    encode_time = time.time() - t0
    full_decoded = tok.decode(full_ids)
    assert full_decoded == normalized
    corpus_cpt = len(text) / len(full_ids)
    print(f"\nFull corpus: {len(text):,} chars → {len(full_ids):,} tokens "
          f"({corpus_cpt:.2f} chars/token, encoded in {encode_time:.1f}s)")

    # --- Save / Load ---
    tok.save(MODEL_DIR)
    loaded = BPETokenizer.load(MODEL_DIR)
    assert loaded.encode(SAMPLE) == ids
    print(f"\nSaved to {MODEL_DIR}/ and reloaded — encode matches ✓")


if __name__ == "__main__":
    main()
