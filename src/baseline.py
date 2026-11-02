"""Step 6 - TF-IDF + Logistic Regression baseline."""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_tfidf_lr(train_texts, y_train, val_texts, y_val, seed):
    """Fit TF-IDF (1-2 grams) + LogisticRegression; pick C on the validation set.

    Inputs are the SAME cleaned/tokenised strings the LSTM sees (but not truncated).
    Returns (predict_proba_fn, seconds, n_parameters, best_C).
    """
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                          sublinear_tf=True)
    Xtr, Xva = vec.fit_transform(train_texts), vec.transform(val_texts)
    best = (-1, None, None)
    for C in (0.5, 2, 8):
        clf = LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=seed)
        clf.fit(Xtr, y_train)
        acc = clf.score(Xva, y_val)
        print(f"[tfidf_lr] C={C}: val acc {acc:.4f}")
        if acc > best[0]:
            best = (acc, C, clf)
    seconds = time.perf_counter() - t0
    _, C, clf = best
    n_params = clf.coef_.size + clf.intercept_.size
    return (lambda texts: clf.predict_proba(vec.transform(texts))[:, 1]), seconds, n_params, C
