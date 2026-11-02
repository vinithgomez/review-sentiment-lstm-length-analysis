"""Central configuration: paths, seeds and hyper-parameters.

Everything that can change an experimental result lives here, so the whole
pipeline can be explained (and reproduced) from one file.
"""
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    quick: bool = False          # tiny smoke-test run, written to runs/quick
    seed: int = 42

    # ---- data -------------------------------------------------------------
    hf_dataset: str = "fancyzhx/amazon_polarity"   # Amazon product reviews, binary
    n_reviews: int = 60_000                        # balanced subsample (CPU friendly)
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ---- text -> ids ------------------------------------------------------
    vocab_size: int = 20_000     # incl. <PAD> and <UNK>
    min_freq: int = 2
    max_len_percentile: float = 95.0   # max_len is picked from the length distribution

    # ---- neural models ----------------------------------------------------
    emb_dim: int = 100           # = GloVe dimension, so both embeddings are comparable
    hidden: int = 128
    dropout: float = 0.4
    lr: float = 2e-3
    batch_size: int = 128
    max_epochs: int = 6
    patience: int = 2            # early stopping on validation loss
    clip: float = 1.0
    threads: int = 4

    # ---- statistics -------------------------------------------------------
    n_boot: int = 2000

    def __post_init__(self):
        if self.quick:
            self.n_reviews, self.max_epochs, self.n_boot = 6_000, 2, 300
        self.base = ROOT / "runs" / "quick" if self.quick else ROOT
        self.data_dir = ROOT / "data"
        self.results = self.base / "results"
        self.figures = self.base / "figures"
        self.models = self.base / "models"
        for d in (self.data_dir, self.results, self.figures, self.models):
            d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, threads: int = 4):
    """Seed every RNG we use so a re-run gives the same numbers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
