"""Step 4 - training loop with early stopping + best-checkpoint saving."""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def iterate(X, L, y, batch_size, shuffle, rng):
    """Mini-batches, each trimmed to the longest real review in the batch (saves CPU)."""
    order = rng.permutation(len(y)) if shuffle else np.arange(len(y))
    for i in range(0, len(order), batch_size):
        b = order[i:i + batch_size]
        m = L[b].max()
        yield (torch.from_numpy(X[b, :m]), torch.from_numpy(L[b]),
               torch.from_numpy(y[b]).float())


@torch.no_grad()
def predict_proba(model, X, L, batch_size=512):
    model.eval()
    out = []
    for xb, lb, _ in iterate(X, L, np.zeros(len(L)), batch_size, False, None):
        out.append(torch.sigmoid(model(xb, lb)).numpy())
    return np.concatenate(out)


def evaluate_loss(model, X, L, y, loss_fn, batch_size=512):
    p = predict_proba(model, X, L, batch_size)
    pt = torch.from_numpy(np.clip(p, 1e-7, 1 - 1e-7)).double()
    loss = nn.functional.binary_cross_entropy(pt, torch.from_numpy(y).double()).item()
    return loss, float(((p >= 0.5) == y).mean())


def fit(model, data, cfg, name, ckpt_path):
    """Train with Adam + BCE-with-logits; stop when validation loss stops improving.

    data = dict with X_train,L_train,y_train,X_val,L_val,y_val. Returns (history, seconds).
    """
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    hist, best, bad = [], float("inf"), 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tl, correct, n = 0.0, 0, 0
        for xb, lb, yb in iterate(data["X_train"], data["L_train"], data["y_train"],
                                  cfg.batch_size, True, rng):
            opt.zero_grad()
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)  # tame exploding grads
            opt.step()
            tl += loss.item() * len(yb)
            correct += ((logits > 0).float() == yb).sum().item()
            n += len(yb)
        vl, va = evaluate_loss(model, data["X_val"], data["L_val"], data["y_val"], loss_fn)
        hist.append(dict(epoch=epoch, train_loss=tl / n, train_acc=correct / n,
                         val_loss=vl, val_acc=va))
        print(f"[{name}] epoch {epoch:2d} | train loss {tl / n:.4f} acc {correct / n:.4f} "
              f"| val loss {vl:.4f} acc {va:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), ckpt_path)       # best checkpoint so far
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[{name}] early stopping (no val-loss gain for {cfg.patience} epochs)")
                break
    seconds = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt_path))             # restore best weights
    return pd.DataFrame(hist), seconds
