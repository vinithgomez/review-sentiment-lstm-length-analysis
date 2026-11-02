# Viva notes (one page)

**LSTM in one paragraph.** A recurrent network reads a review one word at a time and keeps a *memory vector*. A plain RNN overwrites that memory at every word, so early words fade (vanishing gradient). An LSTM adds a **cell state** with three gates — *forget* (what to erase), *input* (what to write), *output* (what to expose) — so useful information can survive many steps. Our model: embedding → LSTM(128) → dropout → one sigmoid unit giving P(positive). Packed sequences make it ignore padding.

**Why might an LSTM beat TF-IDF on long text — and did it?** TF-IDF ignores word order beyond 2-grams, so in theory an LSTM can model negation scope and context over long distances. **In our results it did not**: TF-IDF + LogReg scored 89.2 % vs 88.5 % for the GloVe-LSTM, and the baseline was *better* on long reviews (88.8 % vs 87.0 %, significant). Likely reasons: polarity is mostly decided by individual words/phrases; long reviews mix praise and complaints and the LSTM compresses everything into one final vector; the LSTM only sees the first 170 tokens; small data (42k reviews). Be honest about this — the result is a finding, not a failure.

**Metrics.** *Accuracy* = share correct. *Precision* = of reviews predicted positive, share truly positive. *Recall* = of truly positive reviews, share found. *F1* = harmonic mean of precision and recall. *Confusion matrix* = counts of TN, FP, FN, TP. Classes are balanced, so accuracy is meaningful.

## 10 likely questions

1. **Why remove HTML/URLs but keep "not"?** Markup/URLs carry no sentiment; "not good" ≠ "good", so negations are kept and contractions expanded (`didn't` → `did not`).
2. **Why pack sequences?** Reviews have different lengths; packing stops the LSTM processing `<PAD>` so the final state is after the last real word.
3. **Why a train/validation/test split?** Validation is used for early stopping and choosing the main LSTM and C; test is touched once for the final numbers (no leakage). Vocabulary is built from train only.
4. **What does early stopping do?** Stops when validation loss stops improving (patience 2) and restores the best checkpoint, to prevent overfitting — visible in the curves (train acc ↑, val loss ↑).
5. **Trainable vs GloVe embeddings?** GloVe starts with meaningful word vectors from huge text, helps with small data: 88.5 % vs 87.0 % and faster convergence. Both are fine-tuned.
6. **Why did the simple RNN do worst (80.0 %)?** Vanishing gradients: it forgets early words; the LSTM's gates fix this.
7. **How did you define short/long?** Tertiles of training token counts: short ≤ 48, medium 49–92, long ≥ 93 tokens — chosen from the training set so the test set is not used.
8. **How do you know the length effect is real, not noise?** Bootstrap 95 % CIs and chi-square tests: short–long gap is significant for all neural models (p ≤ 0.001) but not for TF-IDF (p = 0.27). Paired McNemar/bootstrap shows LSTM − baseline is −1.8 points on long reviews (p = 0.005).
9. **What are the main errors?** Short: sarcasm, implicit sentiment, mixed verdicts, label noise. Long: mixed opinions over many sentences; the LSTM is unsure (probabilities near 0.5). Truncation affects only 2.8 % of reviews and is not the main cause.
10. **Limitations?** One dataset, one seed, one sample of 60k; length is confounded with mixed content; small unidirectional LSTM; reviews are short compared with real documents. Better next steps: bidirectional LSTM/attention, multiple seeds, a pretrained transformer.

**Numbers to remember (test set, n = 9,000):** LSTM-GloVe 88.5 % acc / F1 0.883; LSTM-trainable 87.0 %; RNN 80.0 %; TF-IDF+LR 89.2 %. Short/long accuracy of the GloVe-LSTM: 90.3 % / 87.0 %.
