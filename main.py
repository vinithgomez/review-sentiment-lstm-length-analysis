"""Run the whole project:  python main.py   (add --quick for a 2-minute smoke test)

Steps map 1:1 to the sections of REPORT.md:
 1 data | 2 clean+tokenize | 3 embeddings | 4 LSTM | 5 evaluation | 6 baselines | 7 length analysis
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import baseline, data, evaluate, length_analysis as la, plots, preprocess as pp
from src.config import Config, set_seed
from src.models import RecurrentClassifier, count_params, load_glove
from src.train import fit, predict_proba

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="small/fast smoke-test run")
args = parser.parse_args()
cfg = Config(quick=args.quick)
set_seed(cfg.seed, cfg.threads)
R, F = cfg.results, cfg.figures
t_start = time.perf_counter()

# ------------------------------------------------------------------ STEP 1
print("\n===== STEP 1: load labelled dataset =====")
df = data.split_data(data.load_reviews(cfg), cfg)
data.summarize(df, cfg)

# ------------------------------------------------------------------ STEP 2
print("===== STEP 2: clean + tokenize =====")
df["tokens"] = df.text.map(pp.tokenize)
df["n_tokens"] = df.tokens.map(len)
df["tokens_str"] = df.tokens.map(" ".join)
print("example raw   :", df.text.iloc[0][:150])
print("example tokens:", df.tokens.iloc[0][:25])
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
vocab = pp.build_vocab(tr.tokens, cfg.vocab_size, cfg.min_freq)
max_len = int(np.ceil(np.percentile(tr.n_tokens, cfg.max_len_percentile) / 10) * 10)
trunc = (te.n_tokens > max_len).mean()
print(f"vocab size {len(vocab):,} | max_len {max_len} (P{cfg.max_len_percentile:.0f} of train "
      f"lengths) | {trunc:.1%} of test reviews are truncated")
print(tr.n_tokens.describe().round(1).to_string())
tr.n_tokens.describe().to_csv(R / "length_stats_train.csv")
groups, (q33, q67) = la.length_groups(tr.n_tokens.values, te.n_tokens.values)
te["group"] = groups
print(f"length groups: short <= {q33:.0f} < medium <= {q67:.0f} < long tokens")
plots.length_distribution(tr.n_tokens.values, tr.label.values, (q33, q67), max_len,
                          F / "length_distribution.png")
D = {}
for name, part in (("train", tr), ("val", va), ("test", te)):
    X, L = pp.encode(part.tokens, vocab, max_len)
    D[f"X_{name}"], D[f"L_{name}"], D[f"y_{name}"] = X, L, part.label.values.astype(np.int64)
json.dump(dict(vocab_size=len(vocab), max_len=max_len, q33=q33, q67=q67,
               test_truncated_frac=float(trunc)), open(R / "preprocessing_info.json", "w"), indent=1)

# ------------------------------------------------------------------ STEP 3
print("\n===== STEP 3: embeddings =====")
glove, cov = load_glove(vocab, cfg.emb_dim, cfg.data_dir / "glove_raw" / "glove.6B.zip", cfg.seed)

# ------------------------------------------------------------------ STEP 4 + 6 (neural)
print("\n===== STEP 4/6: train LSTM (+ GloVe variant) and simple RNN =====")
specs = [("lstm_emb", "lstm", None), ("rnn", "rnn", None)]
if glove is not None:
    specs.insert(1, ("lstm_glove", "lstm", glove))
probs, info, hists, cms = {}, [], {}, {}
for name, cell, init in specs:
    set_seed(cfg.seed, cfg.threads)          # same init seed for every model
    model = RecurrentClassifier(len(vocab), cfg.emb_dim, cfg.hidden, cell, cfg.dropout, init)
    hist, secs = fit(model, D, cfg, name, cfg.models / f"{name}.pt")
    hist.to_csv(R / f"history_{name}.csv", index=False)
    plots.training_curves(hist, name, F / f"training_curves_{name}.png")
    probs[name] = predict_proba(model, D["X_test"], D["L_test"])
    info.append(dict(model=name, train_seconds=secs, parameters=count_params(model),
                     epochs_run=len(hist), best_epoch=int(hist.val_loss.idxmin()) + 1,
                     best_val_acc=hist.val_acc.iloc[hist.val_loss.idxmin()]))

print("\n----- TF-IDF + Logistic Regression -----")
predict_lr, secs, nparam, bestC = baseline.train_tfidf_lr(tr.tokens_str, D["y_train"],
                                                          va.tokens_str, D["y_val"], cfg.seed)
probs["tfidf_lr"] = predict_lr(te.tokens_str)
info.append(dict(model="tfidf_lr", train_seconds=secs, parameters=nparam, epochs_run=np.nan,
                 best_epoch=np.nan, best_val_acc=np.nan))
print(f"tfidf_lr: {secs:.1f}s, {nparam:,} parameters, C={bestC}")
models = list(probs)

# ------------------------------------------------------------------ STEP 5
print("\n===== STEP 5: evaluation on the held-out test set =====")
y = D["y_test"]
rows = []
for m in models:
    pred = (probs[m] >= 0.5).astype(int)
    rows.append(dict(model=m, **evaluate.metrics(y, pred)))
    cms[m] = evaluate.conf_matrix(y, pred)
    txt = evaluate.report_text(y, pred)
    (R / f"classification_report_{m}.txt").write_text(txt)
    plots.confusion(cms[m], plots.LABELS[m], F / f"confusion_matrix_{m}.png")
    print(f"--- {m} ---\n{txt}{cms[m]}\n")
overall = pd.DataFrame(rows).merge(pd.DataFrame(info), on="model")
overall.to_csv(R / "metrics_overall.csv", index=False)
plots.confusion_grid(cms, F / "confusion_matrices_all.png")

# main LSTM = LSTM variant with the best VALIDATION accuracy (never chosen on test data)
lstm_info = overall[overall.model.str.startswith("lstm")]
main = lstm_info.sort_values("best_val_acc").model.iloc[-1]
base = "tfidf_lr"
(R / "main_model.txt").write_text(main)
print(f"main LSTM chosen on validation accuracy: {main}")

# ------------------------------------------------------------------ STEP 6
print("\n===== STEP 6: comparison table =====")
print(overall.round(4).to_string(index=False))
plots.comparison_bars(overall, F / "model_comparison.png")

# ------------------------------------------------------------------ STEP 7
print("\n===== STEP 7: sentence-length analysis =====")
for m in models:
    te[f"{m}_prob"], te[f"{m}_pred"] = probs[m], (probs[m] >= 0.5).astype(int)
te = te.rename(columns={"label": "y_true"})
te.drop(columns=["tokens"]).to_csv(R / "test_predictions.csv", index=False)

gm = la.group_metrics(te, models)
gm.to_csv(R / "metrics_by_length_group.csv", index=False)
print(gm.round(4).to_string(index=False))
plots.group_bars(gm[gm.group != "all"], F / "accuracy_by_length_group.png")

bins = la.bin_accuracy(te, models, la.length_bins(tr.n_tokens.values))
bins.to_csv(R / "accuracy_by_length_bin.csv", index=False)
plots.accuracy_vs_length(bins, (main, base), F / "accuracy_vs_length.png")

acc_ci, gap, vs, change = la.statistics(te, models, main, base, cfg.n_boot, cfg.seed)
acc_ci.to_csv(R / "stats_accuracy_ci_by_group.csv", index=False)
gap.to_csv(R / "stats_short_vs_long_gap.csv", index=False)
vs.to_csv(R / "stats_main_vs_baseline_by_group.csv", index=False)
change.to_csv(R / "stats_advantage_change.csv", index=False)
for title, t in (("accuracy 95% bootstrap CI", acc_ci), ("short vs long gap", gap),
                 (f"{main} vs {base}", vs), ("does the advantage change with length?", change)):
    print(f"\n{title}\n{t.round(4).to_string(index=False)}")

samples, rates = la.error_analysis(te, main, base, max_len, cfg.seed)
samples.to_csv(R / "error_examples.csv", index=False)
rates.to_csv(R / "error_cue_rates.csv", index=False)
print("\nerror cue rates (wrong=True are the main model's errors)\n", rates.to_string(index=False))
with open(R / "error_examples.md", "w", encoding="utf8") as f:
    for _, r in samples.iterrows():
        f.write(f"- **{r.group}** | {r.n_tokens} tokens | true={'pos' if r.y_true else 'neg'} | "
                f"P(pos)={r[f'{main}_prob']:.2f} | baseline predicted "
                f"{'pos' if r[f'{base}_pred'] else 'neg'} | truncated={r.truncated}\n"
                f"  > {r.text[:700]}\n\n")

print(f"\nDONE in {(time.perf_counter() - t_start) / 60:.1f} min. "
      f"Tables -> {R}, figures -> {F}")
