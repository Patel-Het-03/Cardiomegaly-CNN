import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Find threshold that maximizes F1.
    Returns: best_thr, best_f1, precision, recall
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = (0.5, -1, 0, 0)  # thr, f1, prec, rec

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

        if f1 > best[1]:
            best = (thr, f1, precision, recall)

    return best  # (best_thr, best_f1, precision, recall)

def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return auc, fpr, tpr, thr

def compute_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return cm