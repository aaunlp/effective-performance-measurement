"""Module containing utility functions for formatting and processing NER labels."""


def strip_iob(label):
    """Remove 'B-' or 'I-' prefix from a label (if not 'O')."""
    if label == "O":
        return label
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label
