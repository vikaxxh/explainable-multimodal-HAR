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
    class_names: List[str] = None,
    crossing_class_name: str = "Crossing"
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics for research reporting.

    Args:
        y_true: Ground-truth class indices
        y_pred: Predicted class indices
        class_names: Optional human-readable labels (e.g., 8 pedestrian behavior classes)
        crossing_class_name: Name of crossing class for binary intention benchmark

    Returns:
        Dictionary containing overall, weighted, macro, binary crossing, and per-class reports
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": []
        }

    # 1. Overall Accuracy
    acc = float(accuracy_score(y_t, y_p))

    # Determine class labels
    if class_names is not None:
        labels = list(range(len(class_names)))
    else:
        unique_labels = sorted(list(set(y_t.tolist()) | set(y_p.tolist())))
        labels = unique_labels

    # 2. Macro Metrics across all defined classes
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_t, y_p, labels=labels, average="macro", zero_division=0
    )

    # 3. Weighted Metrics (Support-Weighted - critical for long-tailed multi-class distributions)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0
    )

    # 4. Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=labels)

    report = {
        "accuracy": round(acc * 100.0, 2),
        "macro_precision": round(float(macro_p) * 100.0, 2),
        "macro_recall": round(float(macro_r) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "weighted_precision": round(float(weighted_p) * 100.0, 2),
        "weighted_recall": round(float(weighted_r) * 100.0, 2),
        "weighted_f1": round(float(weighted_f1) * 100.0, 2),
        "confusion_matrix": cm.tolist()
    }

    # 5. Per-Class Metrics
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        y_t, y_p, labels=labels, average=None, zero_division=0
    )

    report["per_class"] = {}
    active_f1_list = []

    names = class_names if class_names is not None else [f"Class_{i}" for i in labels]
    for idx, name in enumerate(names):
        if idx < len(per_p):
            sup = int(per_s[idx])
            f1_val = round(float(per_f[idx]) * 100.0, 2)
            report["per_class"][name] = {
                "precision": round(float(per_p[idx]) * 100.0, 2),
                "recall": round(float(per_r[idx]) * 100.0, 2),
                "f1": f1_val,
                "support": sup
            }
            if sup > 0:
                active_f1_list.append(f1_val)

    # Macro-F1 over active classes only (classes with support > 0)
    if active_f1_list:
        report["active_macro_f1"] = round(float(np.mean(active_f1_list)), 2)
    else:
        report["active_macro_f1"] = report["macro_f1"]

    # 6. Binary Crossing Intention Benchmark (Crossing vs. Non-Crossing)
    crossing_idx = None
    if class_names and crossing_class_name in class_names:
        crossing_idx = class_names.index(crossing_class_name)
    elif 5 in labels:  # Default index for Crossing in EMIT taxonomy
        crossing_idx = 5

    if crossing_idx is not None:
        bin_true = (y_t == crossing_idx).astype(int)
        bin_pred = (y_p == crossing_idx).astype(int)
        bin_acc = float(accuracy_score(bin_true, bin_pred))
        bin_p, bin_r, bin_f, _ = precision_recall_fscore_support(
            bin_true, bin_pred, average="binary", zero_division=0
        )
        report["binary_crossing"] = {
            "accuracy": round(bin_acc * 100.0, 2),
            "precision": round(float(bin_p) * 100.0, 2),
            "recall": round(float(bin_r) * 100.0, 2),
            "f1": round(float(bin_f) * 100.0, 2),
            "crossing_count": int(np.sum(bin_true)),
            "non_crossing_count": int(len(bin_true) - np.sum(bin_true))
        }

    return report
