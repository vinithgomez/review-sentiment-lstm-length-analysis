"""Step 3 + 4 (architecture) - embeddings and the recurrent classifiers."""
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentClassifier(nn.Module):
    """Embedding -> (LSTM | simple RNN) -> dropout -> Linear -> (sigmoid in the loss).

    Variable-length reviews are handled with *packed sequences*: the recurrent layer
    never sees the <PAD> positions, and the final hidden state is the state after the
    last REAL token of every review.
    """

    def __init__(self, vocab_size, emb_dim, hidden, cell="lstm", dropout=0.4,
                 pretrained=None):
        super().__init__()
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:   # initialise with GloVe, still fine-tuned
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
            self.embedding.weight.data[0].zero_()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(emb_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.embedding(x), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        last = h[0][-1] if self.cell == "lstm" else h[-1]     # (batch, hidden)
        return self.fc(self.dropout(last)).squeeze(1)         # logit; sigmoid(logit)=P(positive)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_glove(vocab: dict, emb_dim: int, glove_zip: Path, seed: int):
    """Build an embedding matrix from GloVe 6B (plain-text file inside the Stanford zip).

    Words missing from GloVe get small random vectors. Returns (matrix, coverage) or
    (None, 0) when the file is not available.
    """
    if not glove_zip.exists():           # download once (~860 MB) from the Stanford GloVe repo
        try:
            from huggingface_hub import hf_hub_download
            print("[glove] downloading glove.6B.zip (~860 MB, one-off) ...")
            hf_hub_download("stanfordnlp/glove", "glove.6B.zip", local_dir=str(glove_zip.parent))
        except Exception as e:
            print(f"[glove] download failed ({e}) -> skipping pretrained embeddings")
            return None, 0.0
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 0.1, (len(vocab), emb_dim)).astype(np.float32)
    mat[0] = 0
    hit = 0
    with zipfile.ZipFile(glove_zip) as z, z.open(f"glove.6B.{emb_dim}d.txt") as f:
        for line in f:
            word, _, rest = line.decode("utf8").partition(" ")
            i = vocab.get(word)
            if i is not None and i > 1:
                mat[i] = np.asarray(rest.split(), dtype=np.float32)
                hit += 1
    cov = hit / (len(vocab) - 2)
    print(f"[glove] covered {hit:,}/{len(vocab) - 2:,} vocabulary words ({cov:.1%})")
    return mat, cov
