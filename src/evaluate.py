"""
Phase 4: Evaluation — metrics computation, confusion matrices, and result visualization.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (1D array of class indices).
        y_pred: Predicted labels (1D array of class indices).
    Returns:
        Dict with accuracy, precision, recall, f1 (macro-averaged).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_classification_report(y_true, y_pred, target_name=""):
    """Print a detailed sklearn classification report."""
    label_names = ["Low (0)", "Medium (1)", "High (2)"]
    print(f"\n{'=' * 50}")
    print(f"Classification Report — {target_name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def get_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred, target_name="", save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_name: Name of the target (e.g., 'Valence').
        save_path: Path to save the figure. If None, displays with plt.show().
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available. Skipping plot.")
        return

    cm = confusion_matrix(y_true, y_pred)
    label_names = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {target_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def print_results_table(results_dict):
    """
    Print a formatted results table from LOSOCV results.

    Args:
        results_dict: Dict mapping target names to lists of per-subject dicts
                      (each with accuracy, precision, recall, f1).
    """
    for target, subject_results in results_dict.items():
        print(f"\n{'=' * 70}")
        print(f"  LOSOCV Results — {target.upper()}")
        print(f"{'=' * 70}")
        print(f"{'Subject':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 70)
        for i, res in enumerate(subject_results):
            print(
                f"{'S' + str(i):>10} "
                f"{res['accuracy']:>10.4f} "
                f"{res['precision']:>10.4f} "
                f"{res['recall']:>10.4f} "
                f"{res['f1']:>10.4f}"
            )
        # Average
        avg = {k: np.mean([r[k] for r in subject_results]) for k in subject_results[0]}
        std = {k: np.std([r[k] for r in subject_results]) for k in subject_results[0]}
        print("-" * 70)
        print(
            f"{'MEAN':>10} "
            f"{avg['accuracy']:>10.4f} "
            f"{avg['precision']:>10.4f} "
            f"{avg['recall']:>10.4f} "
            f"{avg['f1']:>10.4f}"
        )
        print(
            f"{'STD':>10} "
            f"{std['accuracy']:>10.4f} "
            f"{std['precision']:>10.4f} "
            f"{std['recall']:>10.4f} "
            f"{std['f1']:>10.4f}"
        )
