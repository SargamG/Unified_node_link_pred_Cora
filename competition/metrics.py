from __future__ import annotations

from sklearn.metrics import f1_score, roc_auc_score


def node_macro_f1(y_true, y_pred_labels) -> float:
    return float(f1_score(y_true, y_pred_labels, average="macro"))


def link_roc_auc(y_true, y_pred_proba) -> float:
    return float(roc_auc_score(y_true, y_pred_proba))

