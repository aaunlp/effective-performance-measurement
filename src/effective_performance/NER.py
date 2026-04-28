import argparse
from collections import Counter
import csv
import datetime
from functools import partial
from pathlib import Path

from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from .eval import create_grouped_error_matrix, run_all_evals
from .formatting import strip_iob
from .model_dataset import (
    convert_dataset_labels,
    convert_model_to_dataset,
    make_dataset_id2label_label2id,
)
from .no_bio import process_dataset, tokenize_and_align_labels


DEFAULT_MODELS = ["rasmus-aau/finer_sequence", "AAU-NLP/BERT-SL1000"]
HOSTED_SECB_DATASET = "AAU-NLP/effective-performance-measurement"
PARQUET_FALLBACK_REVISION = "refs/convert/parquet"
DEFAULT_DATASETS = ["AAU-NLP/HiFi-KPI", "nlpaueb/finer-139", HOSTED_SECB_DATASET]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"


def strip_gaap(label):
    """Remove 'us-gaap:' prefix from label if present."""
    return label[len("us-gaap:") :] if label.startswith("us-gaap:") else label


def normalize_secb_entity(entity):
    if isinstance(entity, list):
        return {
            "Start character": entity[0],
            "End character": entity[1],
            "Label": entity[2],
            "Start date for period": entity[4] if len(entity) > 4 else None,
            "End date for period": entity[5] if len(entity) > 5 else None,
            "Currency / Unit": entity[6] if len(entity) > 6 else None,
            "Value": entity[7] if len(entity) > 7 else None,
        }

    if not isinstance(entity, dict):
        raise TypeError(f"Unsupported entity format: {type(entity)!r}")

    return {
        "Start character": entity.get("Start character", entity.get("start_character")),
        "End character": entity.get("End character", entity.get("end_character")),
        "Label": entity.get("Label", entity.get("label")),
        "Start date for period": entity.get(
            "Start date for period", entity.get("start_date_for_period")
        ),
        "End date for period": entity.get("End date for period", entity.get("end_date_for_period")),
        "Currency / Unit": entity.get(
            "Currency / Unit",
            entity.get("currency_/_unit", entity.get("currency/unit", entity.get("currency_unit"))),
        ),
        "Value": entity.get("Value", entity.get("value")),
    }


def normalize_secb_record(record):
    entities = record.get("entities") or []
    return {
        **record,
        "entities": [normalize_secb_entity(entity) for entity in entities],
    }


def load_dataset_with_parquet_fallback(dataset_name):
    try:
        return load_dataset(dataset_name, download_mode="force_redownload")
    except Exception as error:
        error_text = str(error)
        if "dtype 'int64'" not in error_text and "Invalid value" not in error_text:
            raise

        print(
            f"Raw dataset load for {dataset_name} failed with Arrow type inference. "
            f"Retrying Parquet conversion branch '{PARQUET_FALLBACK_REVISION}'..."
        )
        return load_dataset(
            dataset_name,
            download_mode="force_redownload",
            revision=PARQUET_FALLBACK_REVISION,
        )


def load_processed_dataset(dataset_name, subsample=None):
    if dataset_name == "nlpaueb/finer-139":
        dataset = load_dataset(
            dataset_name,
            download_mode="force_redownload",
            revision=PARQUET_FALLBACK_REVISION,
        )
        ner_feature = dataset["train"].features["ner_tags"]
        label_names = ner_feature.feature.names
        finer_id2label = {i: label for i, label in enumerate(label_names)}
        hf_dataset = dataset["test"]
        new_data = {col: hf_dataset[col] for col in hf_dataset.column_names if col != "ner_tags"}
        new_data["labels"] = [
            [finer_id2label[tag_id] for tag_id in example]
            for example in hf_dataset["ner_tags"]
        ]
        new_hf_dataset = Dataset.from_dict(new_data)

        if subsample is not None:
            return new_hf_dataset.select(range(subsample))
        return new_hf_dataset

    if dataset_name == HOSTED_SECB_DATASET:
        try:
            dataset = load_dataset(
                HOSTED_SECB_DATASET,
                data_files={"test": "SECB.json"},
                download_mode="force_redownload",
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load hosted dataset '{HOSTED_SECB_DATASET}'"
            ) from error

        hf_dataset = dataset["test"]
        print(f"Processing {dataset_name} dataset...")
        if subsample is not None:
            hf_dataset = hf_dataset.select(range(subsample))
        return process_dataset([normalize_secb_record(record) for record in hf_dataset])

    dataset = load_dataset_with_parquet_fallback(dataset_name)
    hf_dataset = dataset["test"]
    print(f"Processing {dataset_name} dataset...")
    if subsample is not None:
        return process_dataset(hf_dataset.select(range(subsample)))
    return process_dataset(hf_dataset)


def run_model_dataset_combination(
    modelName,
    datasetName,
    batch_size,
    subsample=None,
    output_dir=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {modelName} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForTokenClassification.from_pretrained(modelName).to(device)
    model_label2id = model.config.label2id
    model_id2label = model.config.id2label

    print(f"Loading dataset {datasetName}...")
    processed_data = load_processed_dataset(datasetName, subsample=subsample)

    model_label2id = {strip_gaap(strip_iob(label)): id for label, id in model_label2id.items()}
    model_id2label = {id: strip_gaap(strip_iob(label)) for id, label in model_id2label.items()}

    dataset_id2label, dataset_label2id = make_dataset_id2label_label2id(processed_data)
    print(f"Converting {datasetName} data...")

    dataset_id2label = {id: strip_gaap(strip_iob(label)) for label, id in dataset_label2id.items()}
    dataset_label2id = {strip_gaap(strip_iob(label)): id for label, id in dataset_label2id.items()}
    hf_dataset = convert_dataset_labels(processed_data, dataset_label2id)
    print("Tokenizing dataset...")
    tokenized_datasets = hf_dataset.map(
        partial(tokenize_and_align_labels, tokenizer),
        batched=True,
        remove_columns=hf_dataset.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        tokenized_datasets,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    print("Running evaluation...")
    raw_predictions = []
    raw_labels = []

    model.eval()
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        raw_predictions.extend(preds.cpu().tolist())
        raw_labels.extend(labels.cpu().tolist())

    print("Converting predictions to Dataset label space...")
    print()

    special_oss_label = 1000
    all_predictions = [
        [
            "OOS"
            if p == special_oss_label
            else dataset_id2label[convert_model_to_dataset(p, dataset_label2id, model_id2label)]
            for p, l in zip(_pred, lab)
            if l != -100
        ]
        for _pred, lab in zip(raw_predictions, raw_labels)
    ]
    all_true_labels = [
        [dataset_id2label.get(l) for l in lab if l != -100]
        for lab in raw_labels
    ]

    model2idconverted = [
        [model_id2label.get(x) for x in predictions] for predictions in raw_predictions
    ]
    min_len = min(len(raw_predictions), len(all_predictions), len(raw_labels), len(all_true_labels))

    debug_data = []

    for i in range(min_len):
        debug_data.append(
            {
                "Sentence_Index": i,
                "Raw_Pred_IDs": str(raw_predictions[i]),
                "model2id_converted": str(model2idconverted[i]),
                "Converted_Preds": str(all_predictions[i]),
                "Raw_Label_IDs": str(raw_labels[i]),
                "Converted_Labels": str(all_true_labels[i]),
            }
        )

    df_debug = pd.DataFrame(debug_data)
    debug_output_path = results_root / "debug_logic_check.csv"
    df_debug.to_csv(debug_output_path, index=False)

    print(f"Saved debug file to '{debug_output_path}'.")

    model_name_last_part = modelName.split("/")[-1]
    dataset_name_last_part = datasetName.split("/")[-1]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = results_root / f"{model_name_last_part}_on_{dataset_name_last_part}"
    results_dir.mkdir(parents=True, exist_ok=True)

    flat_true_labels = [label for seq in all_true_labels for label in seq]
    flat_predictions = [label for seq in all_predictions for label in seq]

    def print_smart_counts(title, data_list, top_n=10):
        counts = Counter(data_list)
        sorted_counts = counts.most_common()

        print(f"--- {title} ---")
        for label, count in sorted_counts[:top_n]:
            print(f"{label}: {count}")

        rest_count = sum(count for _, count in sorted_counts[top_n:])
        if rest_count > 0:
            print(f"Others (Top {top_n + 1}-n): {rest_count}")
        print()

    print_smart_counts("True Label Counts", flat_true_labels)
    print_smart_counts("Prediction Counts", flat_predictions)
    error_matrix_filename = results_dir / f"error_matrix_{timestamp}.png"

    print("Standard Evaluation Metrics:")
    metrics = run_all_evals(flat_true_labels, flat_predictions)

    print("\nCreating Grouped Error Matrix:")
    create_grouped_error_matrix(
        flat_true_labels,
        flat_predictions,
        save_path=error_matrix_filename,
        figsize=(10, 8),
    )

    metrics["model"] = model_name_last_part
    metrics["dataset"] = dataset_name_last_part
    return metrics


def run_all_combinations(
    models=None,
    datasets=None,
    batch_size=256,
    subsample=None,
    output_dir=None,
):
    models = models or DEFAULT_MODELS
    datasets = datasets or DEFAULT_DATASETS
    results_root = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model in models:
        for dataset in datasets:
            print(f"\n=== Running {model} on {dataset} ===\n")

            try:
                metrics = run_model_dataset_combination(
                    modelName=model,
                    datasetName=dataset,
                    batch_size=batch_size,
                    subsample=subsample,
                    output_dir=results_root,
                )
                all_results.append(metrics)
            except Exception as error:
                print(f"Error running {model} on {dataset}: {error}")
                model_name_last_part = model.split("/")[-1]
                dataset_name_last_part = dataset.split("/")[-1]
                all_results.append(
                    {
                        "model": model_name_last_part,
                        "dataset": dataset_name_last_part,
                        "error": str(error),
                        "status": "failed",
                    }
                )
                continue

    results_df = pd.DataFrame(all_results)

    aggregate_results_path = results_root / "ner_evaluation_results.csv"
    results_df.to_csv(aggregate_results_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    print("\n=== All Results ===\n")
    print(results_df)

    return results_df


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the extracted Clean_HiFi_kpi evaluation workflow."
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model name or path. Repeat to evaluate multiple models.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name. Repeat to evaluate multiple datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        help="Optional dataset subsample size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for CSV, NPZ, PNG, and debug outputs.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_all_combinations(
        models=args.models,
        datasets=args.datasets,
        batch_size=args.batch_size,
        subsample=args.subsample,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
