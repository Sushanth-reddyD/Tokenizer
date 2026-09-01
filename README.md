# Tokenizer

A collection of tokenizers built from scratch, for learning.

## Roadmap

1. **BPE** (char-level, Sennrich et al. 2016) — foundation ✅
2. **Byte-level BPE** (`cl100k_base` regex pretok + raw UTF-8 bytes) — in progress
3. **WordPiece** (BERT)
4. **Unigram LM** (SentencePiece default, T5/LLaMA/Gemma)
5. **SentencePiece wrapper** (whitespace-as-token + BPE-dropout)
6. **Byte-level / tokenizer-free** (ByT5, MegaByte-style)

## Structure

```
bpe/               piece 1 — char-level BPE (complete)
bytelevel/         piece 2 — byte-level BPE (in progress)
tests/             pytest suite
data/              downloaded corpora (gitignored)
```

Each piece is self-contained. `bytelevel/` deliberately re-derives the merge
machinery rather than importing from `bpe/`, so either package reads standalone.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bpe/` is pure stdlib. `bytelevel/` needs the `regex` package — the
`cl100k_base` pattern uses Unicode property escapes (`\p{L}`, `\p{N}`) that
Python's stdlib `re` does not support.

## Usage

TBD as we build.
