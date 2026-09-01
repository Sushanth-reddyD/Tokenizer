# Tokenizer

A collection of tokenizers built from scratch, for learning.

## Roadmap

1. **BPE** (char-level, Sennrich et al. 2016) — foundation
2. **Byte-level BPE** (GPT-2/tiktoken style)
3. **WordPiece** (BERT)
4. **Unigram LM** (SentencePiece default, T5/LLaMA/Gemma)
5. **SentencePiece wrapper** (whitespace-as-token + BPE-dropout)
6. **Byte-level / tokenizer-free** (ByT5, MegaByte-style)

## Structure

```
bpe/               char-level BPE (this piece)
tests/             pytest suite
data/              downloaded corpora (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
```

## Usage

TBD as we build.
