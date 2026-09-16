"""
Classification Metrics Suite.
Phase 9 of implementation plan:
Calculates:
- Overall Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1 Score
- Per-Class Metrics and Confusion Matrix
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics for research reporting.

    Args:
        y_true: Ground-truth class indices
        y_pred: Predicted class indices
        class_names: Optional human-readable labels

    Returns:
        Dictionary containing accuracy, precision, recall, macro_f1, and per-class reports
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_t, y_p)

    report = {
        "accuracy": round(acc * 100.0, 2),
        "macro_precision": round(float(precision) * 100.0, 2),
        "macro_recall": round(float(recall) * 100.0, 2),
        "macro_f1": round(float(f1) * 100.0, 2),
        "confusion_matrix": cm.tolist()
    }

    if class_names:
        per_p, per_r, per_f, _ = precision_recall_fscore_support(
            y_t, y_p, average=None, zero_division=0
        )
        report["per_class"] = {}
        for idx, name in enumerate(class_names):
            if idx < len(per_p):
                report["per_class"][name] = {
                    "precision": round(float(per_p[idx]) * 100.0, 2),
                    "recall": round(float(per_r[idx]) * 100.0, 2),
                    "f1": round(float(per_f[idx]) * 100.0, 2)
                }

    return report
