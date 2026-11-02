"""Step 2 - clean, tokenize, build the vocabulary, encode + pad sequences."""
import re
from collections import Counter

import numpy as np
from nltk.tokenize import RegexpTokenizer

PAD, UNK = "<PAD>", "<UNK>"
PAD_ID, UNK_ID = 0, 1

# Contractions are expanded so that negations survive as the separate token "not"
# ("didn't" -> "did not"). Stop-words are NOT removed: "not", "no", "never" carry
# the sentiment ("not good" vs "good").
_CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"),
    (r"\bain't\b", "is not"), (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"),
    (r"'ll\b", " will"), (r"'m\b", " am"), (r"'d\b", " would"), (r"'s\b", ""),
]
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
# Keep letters/digits plus "!" and "?" (strong sentiment cues); drop other symbols.
_TOKENIZER = RegexpTokenizer(r"[a-z]+|\d+|[!?]")


def clean_text(text: str) -> str:
    t = str(text).lower().replace("’", "'").replace("`", "'")
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\d+", " num ", t)           # all numbers -> one token
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str) -> list:
    return _TOKENIZER.tokenize(clean_text(text)) or [UNK]   # never return an empty review


def build_vocab(train_tokens, vocab_size: int, min_freq: int) -> dict:
    """word -> id from the TRAINING set only (no leakage). Ids 0/1 = <PAD>/<UNK>."""
    counts = Counter(w for toks in train_tokens for w in toks)
    words = [w for w, c in counts.most_common(vocab_size - 2) if c >= min_freq]
    return {PAD: PAD_ID, UNK: UNK_ID, **{w: i + 2 for i, w in enumerate(words)}}


def encode(token_lists, vocab: dict, max_len: int):
    """Tokens -> padded id matrix (right padding, truncation keeps the FIRST max_len tokens).

    Returns (X [n, max_len] int64, lengths [n] = real length after truncation).
    """
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    L = np.zeros(len(token_lists), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(w, UNK_ID) for w in toks[:max_len]]
        X[i, :len(ids)], L[i] = ids, len(ids)
    return X, L
