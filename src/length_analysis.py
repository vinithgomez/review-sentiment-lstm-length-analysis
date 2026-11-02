"""Step 7 - does review length change accuracy? Groups, bins, statistics, errors."""
import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import metrics

NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without"}
CONTRAST = {"but", "however", "although", "though", "yet", "except", "unfortunately"}
GROUPS = ["short", "medium", "long"]


def length_groups(train_len, test_len):
    """Tertiles of the TRAINING length distribution -> thresholds fixed without the test set.

    short: n_tokens <= q33 | medium: q33 < n <= q67 | long: n > q67
    """
    q33, q67 = np.quantile(train_len, [1 / 3, 2 / 3])
    g = np.where(test_len <= q33, "short", np.where(test_len <= q67, "medium", "long"))
    return g, (q33, q67)


def length_bins(train_len, n_bins=10):
    """Decile edges of the training lengths (equal-sized bins -> comparable noise)."""
    return np.unique(np.quantile(train_len, np.linspace(0, 1, n_bins + 1)[1:-1]))


def group_metrics(df, models):
    rows = []
    for m in models:
        for g in GROUPS + ["all"]:
            sub = df if g == "all" else df[df.group == g]
            rows.append(dict(model=m, group=g, n=len(sub),
                             **metrics(sub.y_true, sub[f"{m}_pred"])))
    return pd.DataFrame(rows)


def bin_accuracy(df, models, edges):
    b = np.digitize(df.n_tokens, edges)
    rows = []
    for k in np.unique(b):
        sub = df[b == k]
        for m in models:
            rows.append(dict(bin=int(k), n=len(sub), median_len=float(sub.n_tokens.median()),
                             lo=int(sub.n_tokens.min()), hi=int(sub.n_tokens.max()),
                             model=m, accuracy=(sub.y_true == sub[f"{m}_pred"]).mean()))
    return pd.DataFrame(rows)


def _boot(rng, arrs_by_group, n_boot, stat):
    """Bootstrap: resample rows within each group independently, apply `stat`."""
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = stat([a[rng.integers(0, len(a), len(a))] for a in arrs_by_group])
    return out


def statistics(df, models, main, base, n_boot, seed):
    """Bootstrap CIs, McNemar tests and short-vs-long significance tests."""
    rng = np.random.default_rng(seed)
    correct = {m: (df.y_true == df[f"{m}_pred"]).values.astype(float) for m in models}
    grp = {g: (df.group == g).values for g in GROUPS}
    ci = lambda x: tuple(np.percentile(x, [2.5, 97.5]))

    # 1) accuracy CI per model and group
    rows = []
    for m in models:
        for g in GROUPS:
            c = correct[m][grp[g]]
            lo, hi = ci(_boot(rng, [c], n_boot, lambda a: a[0].mean()))
            rows.append(dict(model=m, group=g, accuracy=c.mean(), ci_low=lo, ci_high=hi))
    acc_ci = pd.DataFrame(rows)

    # 2) short vs long accuracy gap per model (independent samples)
    rows = []
    for m in models:
        s, l = correct[m][grp["short"]], correct[m][grp["long"]]
        lo, hi = ci(_boot(rng, [s, l], n_boot, lambda a: a[0].mean() - a[1].mean()))
        table = [[s.sum(), len(s) - s.sum()], [l.sum(), len(l) - l.sum()]]
        chi2, p, _, _ = stats.chi2_contingency(table)
        rows.append(dict(model=m, acc_short=s.mean(), acc_long=l.mean(),
                         diff_short_minus_long=s.mean() - l.mean(), ci_low=lo, ci_high=hi,
                         chi2_p_value=p))
    gap = pd.DataFrame(rows)

    # 3) main model vs baseline: paired bootstrap + McNemar per group
    rows = []
    for g in GROUPS + ["all"]:
        mask = np.ones(len(df), bool) if g == "all" else grp[g]
        a, b = correct[main][mask], correct[base][mask]
        pair = np.stack([a, b], 1)
        lo, hi = ci(_boot(rng, [pair], n_boot, lambda x: (x[0][:, 0] - x[0][:, 1]).mean()))
        only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append(dict(group=g, n=int(mask.sum()), acc_main=a.mean(), acc_base=b.mean(),
                         diff_main_minus_base=a.mean() - b.mean(), ci_low=lo, ci_high=hi,
                         main_only_correct=only_a, base_only_correct=only_b,
                         mcnemar_p_value=p))
    vs = pd.DataFrame(rows)

    # 4) does the main-vs-baseline advantage change between short and long?
    pairs = {g: np.stack([correct[main][grp[g]], correct[base][grp[g]]], 1)
             for g in ("short", "long")}
    d = lambda p: (p[:, 0] - p[:, 1]).mean()
    obs = d(pairs["long"]) - d(pairs["short"])
    boots = _boot(rng, [pairs["short"], pairs["long"]], n_boot, lambda x: d(x[1]) - d(x[0]))
    lo, hi = ci(boots)
    change = pd.DataFrame([dict(main=main, base=base,
                                advantage_short=d(pairs["short"]),
                                advantage_long=d(pairs["long"]),
                                change_long_minus_short=obs, ci_low=lo, ci_high=hi)])
    return acc_ci, gap, vs, change


def tag_cues(text_tokens):
    s = set(text_tokens)
    return dict(has_negation=bool(s & NEGATIONS), has_contrast=bool(s & CONTRAST),
                has_exclaim="!" in s)


def error_analysis(df, main, base, max_len, seed, k=8):
    """Sample k misclassified short and long reviews + cue rates (errors vs correct)."""
    d = df.copy()
    d["wrong"] = d.y_true != d[f"{main}_pred"]
    cues = pd.DataFrame([tag_cues(t.split()) for t in d.tokens_str], index=d.index)
    d = pd.concat([d, cues], axis=1)
    d["truncated"] = d.n_tokens > max_len
    d["base_wrong"] = d.y_true != d[f"{base}_pred"]
    cols = ["has_negation", "has_contrast", "has_exclaim", "truncated"]
    rate = (d[d.group.isin(["short", "long"])]
            .groupby(["group", "wrong"])[cols + ["n_tokens"]]
            .mean().round(3).reset_index())
    rate.insert(2, "n", d[d.group.isin(["short", "long"])].groupby(["group", "wrong"]).size().values)
    samples = []
    for g in ("short", "long"):
        pool = d[(d.group == g) & d.wrong]
        samples.append(pool.sample(min(k, len(pool)), random_state=seed))
    samples = pd.concat(samples)
    samples = samples[["group", "n_tokens", "y_true", f"{main}_prob", f"{base}_pred",
                       "truncated", "has_negation", "has_contrast", "text"]]
    return samples, rate
