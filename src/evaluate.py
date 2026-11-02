"""Step 5 - metrics, classification report, confusion matrix."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def metrics(y_true, y_pred) -> dict:
    """Accuracy + precision/recall/F1 of the POSITIVE class (label 1)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f)


def report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["negative", "positive"],
                                 digits=4)


def conf_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
