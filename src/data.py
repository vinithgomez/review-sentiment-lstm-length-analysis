"""Step 1 - load a labelled review dataset, balance it, split it."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def load_reviews(cfg: Config) -> pd.DataFrame:
    """Return a balanced DataFrame with columns text,label (1 = positive, 0 = negative).

    Source: Amazon Polarity (Hugging Face). Reviews rated 1-2 stars are labelled
    negative and 4-5 stars positive; 3-star (neutral) reviews are already removed
    by the dataset authors. Fallback: put your own data/custom.csv (text,label).
    """
    custom = cfg.data_dir / "custom.csv"
    cache = cfg.data_dir / f"reviews_{cfg.n_reviews}_seed{cfg.seed}.csv"
    if custom.exists():
        df = pd.read_csv(custom)[["text", "label"]].dropna()
        print(f"[data] using user-supplied {custom}")
    elif cache.exists():
        df = pd.read_csv(cache)
        print(f"[data] loaded cached sample {cache.name}")
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(cfg.hf_dataset, split="train")
        except Exception as e:  # no internet etc.
            raise SystemExit(
                f"Dataset download failed ({e}).\nPlace a CSV with columns 'text','label' "
                f"(1=positive, 0=negative) at {custom} and re-run.")
        print(f"[data] full dataset: {len(ds):,} reviews")
        labels = np.asarray(ds["label"])
        rng = np.random.default_rng(cfg.seed)
        per_class = cfg.n_reviews // 2
        # Draw the same number of reviews from each class -> perfectly balanced.
        idx = np.concatenate([rng.choice(np.where(labels == c)[0], per_class, replace=False)
                              for c in (0, 1)])
        sub = ds.select(np.sort(idx))
        df = pd.DataFrame({"text": sub["content"], "label": sub["label"]})
        df.to_csv(cache, index=False)
    df = df[df.text.astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, cfg: Config):
    """Stratified 70/15/15 train/validation/test split (index-aligned columns)."""
    idx = np.arange(len(df))
    trval, test = train_test_split(idx, test_size=cfg.test_frac, stratify=df.label,
                                   random_state=cfg.seed)
    train, val = train_test_split(trval, test_size=cfg.val_frac / (1 - cfg.test_frac),
                                  stratify=df.label.values[trval], random_state=cfg.seed)
    df = df.copy()
    df["split"] = ""
    df.loc[train, "split"], df.loc[val, "split"], df.loc[test, "split"] = "train", "val", "test"
    return df


def summarize(df: pd.DataFrame, cfg: Config):
    """Print + save dataset size, class balance and sample rows."""
    bal = df.groupby(["split", "label"]).size().unstack()
    bal.columns = ["negative", "positive"]
    print(f"[data] {len(df):,} reviews used\n{bal}\n")
    bal.to_csv(cfg.results / "dataset_class_balance.csv")
    samples = df.groupby("label").sample(3, random_state=cfg.seed)[["label", "text"]]
    samples["text"] = samples.text.str.slice(0, 160)
    print(samples.to_string(index=False), "\n")
    samples.to_csv(cfg.results / "dataset_samples.csv", index=False)
