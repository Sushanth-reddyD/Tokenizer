"""WordPiece tokenizer (BERT lineage).

Where BPE merges the most *frequent* pair, WordPiece merges the pair that
maximizes a likelihood score: freq(ab) / (freq(a) * freq(b)).  And where
BPE encodes by replaying merges in rank order, WordPiece uses greedy
longest-prefix matching against the vocab.

Built from scratch for learning.
"""

from .pretokenizer import pretokenize
from .tokenizer import WordPieceTokenizer
