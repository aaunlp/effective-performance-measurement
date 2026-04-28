from datasets import Dataset

from .formatting import strip_iob


def strip_gaap(label):
    """Remove 'us-gaap:' prefix from label if present."""
    return label[len("us-gaap:") :] if label.startswith("us-gaap:") else label


def convert_model_to_dataset(tag_id, dataset_label2id, model_id2label):
    """Convert model prediction ID to dataset label ID."""
    if tag_id not in model_id2label:
        print("UNK LABEL!")
        return dataset_label2id.get("UNK")
    model_label = model_id2label[tag_id]

    if model_label in dataset_label2id:
        mapped_label = model_label
    else:
        mapped_label = "UNK"
    return dataset_label2id.get(mapped_label, dataset_label2id.get("UNK"))


def convert_dataset_labels(processed_data, dataset_label2id):
    """
    Convert processed data to Hugging Face dataset format with label reformatting.

    Args:
        processed_data (list): List of processed data items
        dataset_label2id (dict): Mapping from labels to IDs

    Returns:
        Dataset: Hugging Face dataset object
    """
    default_id = dataset_label2id.get("UNK", 0)

    def process_label(label):
        return dataset_label2id.get(strip_gaap(strip_iob(label)), default_id)

    converted_dataset = [
        {
            "ner_tags": [process_label(label) for label in item["labels"]],
            "tokens": item["tokens"],
        }
        for item in processed_data
    ]

    return Dataset.from_list(converted_dataset)


def make_dataset_id2label_label2id(processed_data):
    dataset_labels = set()
    for rec in processed_data:
        dataset_labels.update(rec["labels"])
    dataset_labels = sorted(list(dataset_labels))

    dataset_label2id = {label: idx for idx, label in enumerate(dataset_labels)}
    dataset_id2label = {idx: label for label, idx in dataset_label2id.items()}
    if "UNK" not in dataset_label2id:
        dataset_label2id["UNK"] = len(dataset_label2id)
        dataset_id2label[len(dataset_id2label)] = "UNK"
    return dataset_id2label, dataset_label2id
