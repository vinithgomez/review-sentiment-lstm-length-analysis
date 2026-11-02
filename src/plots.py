"""All figures. Colour-blind-safe palette (Okabe-Ito), one colour per model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LABELS = {"lstm_emb": "LSTM (trainable emb.)", "lstm_glove": "LSTM (GloVe init)",
          "rnn": "Simple RNN", "tfidf_lr": "TF-IDF + LogReg"}
COLORS = {"lstm_emb": "#0072B2", "lstm_glove": "#56B4E9", "rnn": "#E69F00",
          "tfidf_lr": "#D55E00"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def length_distribution(lengths, labels, cuts, max_len, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, name, c in ((0, "negative", "#D55E00"), (1, "positive", "#0072B2")):
        ax.hist(lengths[labels == lab], bins=60, alpha=0.55, label=name, color=c)
    for q, nm in zip(cuts, ("33rd pct", "67th pct")):
        ax.axvline(q, color="k", ls="--", lw=1)
        ax.text(q, ax.get_ylim()[1] * 0.95, f" {nm}={q:.0f}", fontsize=8, va="top")
    ax.axvline(max_len, color="grey", ls=":", lw=1.5)
    ax.text(max_len, ax.get_ylim()[1] * 0.8, f" max_len={max_len}", fontsize=8, color="grey")
    ax.set(xlabel="review length (tokens)", ylabel="number of reviews",
           title="Review length distribution (training set)")
    ax.legend()
    _save(fig, path)


def training_curves(hist, name, path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, yl in zip(axs, ("loss", "acc"), ("loss", "accuracy")):
        ax.plot(hist.epoch, hist[f"train_{key}"], "o-", label="train", color="#0072B2")
        ax.plot(hist.epoch, hist[f"val_{key}"], "s-", label="validation", color="#D55E00")
        ax.set(xlabel="epoch", ylabel=yl, title=f"{LABELS[name]} - {yl}")
        ax.set_xticks(hist.epoch)
        ax.legend()
    _save(fig, path)


def confusion(cm, title, path=None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    ax.set(xlabel="predicted", ylabel="true", title=title)
    ax.grid(False)
    if own:
        _save(fig, path)


def confusion_grid(cms, path):
    fig, axs = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.6))
    for ax, (m, cm) in zip(np.atleast_1d(axs), cms.items()):
        confusion(cm, LABELS[m], ax=ax)
    _save(fig, path)


def comparison_bars(df, path):
    """df: model, accuracy, precision, recall, f1 (one row per model)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    mets, w = ["accuracy", "precision", "recall", "f1"], 0.2
    for i, m in enumerate(df.model):
        vals = df.set_index("model").loc[m, mets].values.astype(float)
        bars = ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, label=LABELS[m], color=COLORS[m])
        ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(range(4), [m.upper() if m == "f1" else m.capitalize() for m in mets])
    lo = max(0, df[mets].values.min() - 0.05)
    ax.set(ylim=(lo, min(1.0, df[mets].values.max() + 0.05)), ylabel="score",
           title=f"Test-set comparison (y-axis starts at {lo:.2f})")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def group_bars(gm, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = ["short", "medium", "long"]
    models = list(gm.model.unique())
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [gm[(gm.model == m) & (gm.group == g)].accuracy.item() for g in groups]
        bars = ax.bar(np.arange(3) + (i - (len(models) - 1) / 2) * w, v, w,
                      label=LABELS[m], color=COLORS[m])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(range(3), groups)
    lo = max(0, gm.accuracy.min() - 0.05)
    ax.set(ylim=(lo, min(1.0, gm.accuracy.max() + 0.04)), ylabel="accuracy",
           title=f"Accuracy by review-length group (y-axis starts at {lo:.2f})")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def accuracy_vs_length(bins, gap_pair, path):
    """Left: accuracy per length decile per model. Right: LSTM minus baseline."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for m in bins.model.unique():
        s = bins[bins.model == m]
        axs[0].plot(s.median_len, s.accuracy, "o-", label=LABELS[m], color=COLORS[m])
    axs[0].set(xlabel="review length (median tokens in bin)", ylabel="accuracy",
               title="Accuracy vs review length (10 equal-size bins)")
    axs[0].legend(fontsize=8)
    main, base = gap_pair
    a, b = bins[bins.model == main], bins[bins.model == base]
    axs[1].bar(a.median_len.values.astype(str), (a.accuracy.values - b.accuracy.values) * 100,
               color=COLORS[main])
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set(xlabel="review length (median tokens in bin)",
               ylabel="accuracy difference (percentage points)",
               title=f"{LABELS[main]} minus {LABELS[base]}")
    _save(fig, path)
