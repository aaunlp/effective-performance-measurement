"""Module containing evaluation metrics and functions for NER model performance assessment."""

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_all_evals(flat_true_labels, flat_predictions):
    """
    Run all evaluation metrics on the predicted and true labels.

    Args:
        flat_true_labels: List of true labels
        flat_predictions: List of predicted labels

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    flat_true_labels = np.array(flat_true_labels)
    flat_predictions = np.array(flat_predictions)

    non_o_indices = [i for i, label in enumerate(flat_true_labels) if label != "O"]

    filtered_true_labels = flat_true_labels[non_o_indices]
    filtered_predictions = flat_predictions[non_o_indices]

    metrics = {}
    metrics["f1_macro"] = f1_score(filtered_true_labels, filtered_predictions, average="macro")
    metrics["f1_weighted"] = f1_score(filtered_true_labels, filtered_predictions, average="weighted")
    metrics["f1_micro"] = f1_score(filtered_true_labels, filtered_predictions, average="micro")
    metrics["accuracy"] = accuracy_score(filtered_true_labels, filtered_predictions)
    metrics["precision_weighted"] = precision_score(
        filtered_true_labels, filtered_predictions, average="weighted"
    )
    metrics["precision_micro"] = precision_score(
        filtered_true_labels, filtered_predictions, average="micro"
    )
    metrics["recall_weighted"] = recall_score(
        filtered_true_labels, filtered_predictions, average="weighted"
    )
    metrics["recall_micro"] = recall_score(filtered_true_labels, filtered_predictions, average="micro")
    metrics["kappa"] = cohen_kappa_score(filtered_true_labels, filtered_predictions)

    print(f"macro F1 (ignoring O): {metrics['f1_macro']:.4f}")
    print(f"weighted F1 (ignoring O): {metrics['f1_weighted']:.4f}")
    print(f"micro F1 (ignoring O): {metrics['f1_micro']:.4f}")
    print(f"Accuracy (ignoring O): {metrics['accuracy']:.4f}")
    print(f"weighted Precision (ignoring O): {metrics['precision_weighted']:.4f}")
    print(f"micro Precision (ignoring O): {metrics['precision_micro']:.4f}")
    print(f"weighted Recall (ignoring O): {metrics['recall_weighted']:.4f}")
    print(f"micro Recall (ignoring O): {metrics['recall_micro']:.4f}")
    print(f"Cohen's Kappa (ignoring O): {metrics['kappa']:.4f}")

    return metrics


def create_grouped_error_matrix(flat_true_labels, flat_predictions, save_path=None, figsize=(10, 8)):
    """
    Create and visualize a grouped confusion matrix for NER labels, categorizing them into
    'O_label', 'regex_label', and 'real_label' groups.
    """
    flat_true_labels = np.array(flat_true_labels)
    flat_predictions = np.array(flat_predictions)

    def group_label(label):
        if label == "O":
            return "O_label"
        if "regex_tagged_" in label.lower() or "_tagged_" in label.lower():
            return "regex_label"
        return "real_label"

    grouped_true = np.array([group_label(label) for label in flat_true_labels])
    grouped_pred = np.array([group_label(label) for label in flat_predictions])
    observed_groups = sorted(set(np.concatenate([grouped_true, grouped_pred])))

    preferred_order = ["real_label", "regex_label", "O_label"]
    group_order = [group for group in preferred_order if group in observed_groups]

    cm = confusion_matrix(grouped_true, grouped_pred, labels=group_order)
    grouped_matrix = pd.DataFrame(cm, index=group_order, columns=group_order)

    metrics_by_group = {}
    for group in group_order:
        group_indices = np.where(grouped_true == group)[0]

        if len(group_indices) > 0:
            group_true = grouped_true[group_indices]
            group_pred = grouped_pred[group_indices]

            metrics_by_group[group] = {
                "accuracy": accuracy_score(group_true, group_pred),
                "precision": precision_score(
                    group_true,
                    group_pred,
                    average="micro",
                    labels=[group],
                    zero_division=0,
                ),
                "recall": recall_score(
                    group_true,
                    group_pred,
                    average="micro",
                    labels=[group],
                    zero_division=0,
                ),
                "total_instances": len(group_indices),
                "correct_predictions": sum(group_true == group_pred),
            }

    print("\nSummary by Group:")
    for group, metrics in metrics_by_group.items():
        correct = metrics["correct_predictions"]
        total = metrics["total_instances"]
        print(f"{group}: {correct}/{total} correct ({correct/total:.1%})")

    missing_groups = set(preferred_order) - set(group_order)
    if missing_groups:
        print(f"\nNote: The following groups have no instances in the data: {', '.join(missing_groups)}")

    plt.figure(figsize=figsize)

    row_sums = grouped_matrix.sum(axis=1)
    percentage_matrix = grouped_matrix.astype(float)
    for i, total in enumerate(row_sums):
        if total > 0:
            percentage_matrix.iloc[i] = percentage_matrix.iloc[i] / total * 100
        else:
            percentage_matrix.iloc[i] = 0.0

    annot_matrix = percentage_matrix.astype(object)
    for i in range(len(grouped_matrix.index)):
        for j in range(len(grouped_matrix.columns)):
            count = int(grouped_matrix.iloc[i, j])
            pct = percentage_matrix.iloc[i, j]
            if count > 0:
                annot_matrix.iloc[i, j] = f"{pct:.1f}%\n({count})"
            else:
                annot_matrix.iloc[i, j] = "0.0%"

    display_names = {
        "O_label": "Non-Entity",
        "regex_label": "Regex Tagged",
        "real_label": "XBRL Tagged",
    }

    display_index = [display_names.get(idx, idx) for idx in grouped_matrix.index]
    display_columns = [display_names.get(col, col) for col in grouped_matrix.columns]

    ax = sns.heatmap(
        percentage_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Percentage (%)"},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Percentage (%)", fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xticklabels(display_columns, fontsize=18)
    ax.set_yticklabels(display_index, fontsize=18)
    for text in ax.texts:
        text.set_fontsize(16)

    plt.xlabel("Predicted Label Group", fontsize=22)
    plt.ylabel("True Label Group", fontsize=22)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()

    return grouped_matrix, metrics_by_group
