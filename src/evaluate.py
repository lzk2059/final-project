"""Evaluate trained classifiers and create reproducible result figures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import DataLoaderConfig, InfraredDataset, create_dataloader
from src.model import create_model, get_parameter_counts


SPLIT_FILES = {"validation": "val", "test": "test"}
GPU_WARMUP_ITERATIONS = 5


def validate_class_indices(
    values: pd.Series,
    num_classes: int,
    source: str,
) -> np.ndarray:
    """Reject missing, fractional, and out-of-range class indices."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Invalid class indices found in {source}.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"Non-integer class indices found in {source}.")
    if not ((0 <= numeric) & (numeric < num_classes)).all():
        raise ValueError(
            f"Class indices in {source} must be between 0 and {num_classes - 1}."
        )
    return numeric.astype(np.int64)


def load_class_names(root: Path, num_classes: int) -> list[str]:
    """Load class names in numeric-index order."""

    path = root / "data" / "processed" / "label_to_index.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise TypeError("label_to_index.json must contain a mapping.")
    names = [""] * num_classes
    for name, index in mapping.items():
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_classes
        ):
            raise ValueError(f"Invalid class index for '{name}': {index}")
        if names[index]:
            raise ValueError(f"Duplicate class index: {index}")
        names[index] = str(name)
    if any(not name for name in names):
        raise ValueError("Class mapping does not contain every class index.")
    return names


def checkpoint_experiment_name(path: Path) -> str:
    """Derive a stable experiment name from a best-checkpoint filename."""

    stem = path.stem
    return stem[:-5] if stem.endswith("_best") else stem


def synchronise(device: torch.device) -> None:
    """Wait for queued CUDA work before measuring elapsed time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_deep_model(
    checkpoint_path: Path,
    split: str,
    root: Path,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Load a PyTorch checkpoint and return targets and probabilities."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    num_classes = int(checkpoint["num_classes"])
    model = create_model(model_name, num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_file = SPLIT_FILES[split]
    csv_path = root / "data" / "splits" / f"{split_file}.csv"
    dataset = InfraredDataset(csv_path, project_root=root, training=False)
    loader = create_dataloader(
        dataset,
        DataLoaderConfig(batch_size=batch_size, num_workers=num_workers),
    )
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    inference_seconds = 0.0
    processed_samples = 0
    warmup_iterations = 0

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if max_samples is not None:
                remaining = max_samples - processed_samples
                if remaining <= 0:
                    break
                images, labels = images[:remaining], labels[:remaining]
            images = images.to(device, non_blocking=True)
            if device.type == "cuda" and warmup_iterations == 0:
                for _ in range(GPU_WARMUP_ITERATIONS):
                    model(images)
                synchronise(device)
                warmup_iterations = GPU_WARMUP_ITERATIONS
            synchronise(device)
            start = time.perf_counter()
            logits = model(images)
            synchronise(device)
            inference_seconds += time.perf_counter() - start
            if logits.shape != (len(labels), num_classes):
                raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model produced NaN or Inf logits.")
            targets.append(labels.numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            processed_samples += len(labels)
            if max_samples is not None and processed_samples >= max_samples:
                break
            if batch_index % 50 == 0:
                print(f"Evaluated {processed_samples:,} images")

    y_true = np.concatenate(targets)
    y_probability = np.concatenate(probabilities)
    sample_frame = dataset.dataframe.iloc[: len(y_true)].copy()
    details = {
        "model_type": "deep_learning",
        "model_name": model_name,
        "device": str(device),
        "parameter_counts": get_parameter_counts(model),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_macro_f1": checkpoint.get(
            "best_validation_macro_f1", checkpoint.get("validation_macro_f1")
        ),
        "gpu_warmup_iterations": warmup_iterations,
        "inference_seconds": inference_seconds,
    }
    return y_true, y_probability, sample_frame, details


def evaluate_random_forest(
    checkpoint_path: Path,
    split: str,
    root: Path,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Load a Random Forest checkpoint and evaluate handcrafted features."""

    checkpoint = joblib.load(checkpoint_path)
    model = checkpoint["model"]
    feature_columns = checkpoint["feature_columns"]
    split_file = SPLIT_FILES[split]
    csv_path = root / "data" / "features" / f"{split_file}_features.csv"
    dataframe = pd.read_csv(csv_path)
    missing = set(feature_columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing)}")
    if max_samples is not None:
        dataframe = dataframe.iloc[:max_samples].copy()
    num_classes = len(checkpoint["class_names"])
    targets = validate_class_indices(
        dataframe["class_index"], num_classes, csv_path.name
    )
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature CSV contains NaN or Inf values.")

    start = time.perf_counter()
    partial_probabilities = model.predict_proba(features)
    inference_seconds = time.perf_counter() - start
    probabilities = np.zeros((len(dataframe), num_classes), dtype=np.float64)
    probabilities[:, model.classes_.astype(int)] = partial_probabilities
    details = {
        "model_type": "random_forest",
        "model_name": type(model).__name__,
        "device": "cpu",
        "parameter_counts": None,
        "number_of_trees": int(model.n_estimators),
        "number_of_features": len(feature_columns),
        "best_parameters": checkpoint.get("best_parameters"),
        "inference_seconds": inference_seconds,
    }
    return (
        targets,
        probabilities,
        dataframe,
        details,
    )


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Calculate overall, per-class, and confusion-matrix metrics."""

    labels = list(range(len(class_names)))
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    metrics = {
        "number_of_samples": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
    }
    return metrics, predictions, normalized


def save_predictions(
    path: Path,
    samples: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> None:
    """Save sample identifiers, labels, predictions, and all probabilities."""

    output = pd.DataFrame(
        {
            "image_id": samples["image_id"].to_numpy(),
            "image_path": samples["image_path"].to_numpy(),
            "true_index": targets,
            "true_label": [class_names[index] for index in targets],
            "predicted_index": predictions,
            "predicted_label": [class_names[index] for index in predictions],
            "confidence": probabilities.max(axis=1),
        }
    )
    for index, name in enumerate(class_names):
        output[f"probability_{index}_{name}"] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    path: Path,
    normalized: bool,
) -> None:
    """Save one raw or row-normalized confusion matrix."""

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Normalized Confusion Matrix" if normalized else "Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_per_class_metrics(
    per_class: dict[str, dict[str, float]], path: Path
) -> None:
    """Save grouped bars for per-class precision, recall, and F1."""

    names = list(per_class)
    positions = np.arange(len(names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 7))
    for offset, metric in zip((-width, 0, width), ("precision", "recall", "f1")):
        axis.bar(
            positions + offset,
            [per_class[name][metric] for name in names],
            width,
            label=metric.title(),
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xticks(positions, names, rotation=45, ha="right")
    axis.set_title("Per-class Classification Metrics")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_training_history(history_path: Path, figure_dir: Path, experiment: str) -> None:
    """Generate four independent figures from a saved training history."""

    document = json.loads(history_path.read_text(encoding="utf-8"))
    history = document["history"]
    epochs = [record["epoch"] for record in history]
    best_epoch = max(history, key=lambda item: item["validation"]["macro_f1"])["epoch"]
    definitions = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "learning_rate": "Learning Rate",
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in definitions.items():
        figure, axis = plt.subplots(figsize=(8, 6))
        if metric == "learning_rate":
            default_rate = document.get("config", {}).get("learning_rate")
            rates = [row.get(metric, default_rate) for row in history]
            if any(rate is None for rate in rates):
                plt.close(figure)
                continue
            axis.plot(epochs, rates, marker="o")
        else:
            for split, display in (("train", "Training"), ("validation", "Validation")):
                axis.plot(
                    epochs,
                    [row[split][metric] for row in history],
                    marker="o",
                    label=display,
                )
            axis.legend()
        if metric == "macro_f1":
            axis.axvline(best_epoch, color="tab:red", linestyle="--")
        axis.set(xlabel="Epoch", ylabel=label, title=f"{experiment}: {label}")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(figure_dir / f"{experiment}_{metric}.png", dpi=300)
        plt.close(figure)


def evaluate(
    checkpoint: str | Path,
    split: str = "validation",
    project_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int | None = None,
    history_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate one checkpoint and save metrics, predictions, and figures."""

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    experiment = checkpoint_experiment_name(checkpoint_path)

    if checkpoint_path.suffix.lower() == ".pt":
        targets, probabilities, samples, details = evaluate_deep_model(
            checkpoint_path, split, root, batch_size, num_workers, max_samples
        )
    elif checkpoint_path.suffix.lower() == ".joblib":
        targets, probabilities, samples, details = evaluate_random_forest(
            checkpoint_path, split, root, max_samples
        )
    else:
        raise ValueError("Checkpoint must end in .pt or .joblib.")

    class_names = load_class_names(root, probabilities.shape[1])
    metrics, predictions, normalized_matrix = calculate_metrics(
        targets, probabilities, class_names
    )
    details["inference_seconds_per_image"] = (
        details["inference_seconds"] / len(targets)
    )
    result = {
        "experiment_name": experiment,
        "split": split,
        "debug_evaluation": max_samples is not None,
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "model_details": details,
        "metrics": metrics,
    }

    suffix = f"_{split}" + ("_debug" if max_samples is not None else "")
    evaluation_dir = root / "results" / "evaluation"
    figure_dir = root / "results" / "figures"
    result_path = evaluation_dir / f"{experiment}{suffix}_metrics.json"
    predictions_path = evaluation_dir / f"{experiment}{suffix}_predictions.csv"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_predictions(
        predictions_path,
        samples,
        targets,
        predictions,
        probabilities,
        class_names,
    )
    matrix = np.asarray(metrics["confusion_matrix"])
    plot_confusion_matrix(
        matrix, class_names, figure_dir / f"{experiment}{suffix}_confusion_matrix.png", False
    )
    plot_confusion_matrix(
        normalized_matrix,
        class_names,
        figure_dir / f"{experiment}{suffix}_confusion_matrix_normalized.png",
        True,
    )
    plot_per_class_metrics(
        metrics["per_class"],
        figure_dir / f"{experiment}{suffix}_per_class_metrics.png",
    )

    resolved_history = Path(history_path).expanduser().resolve() if history_path else (
        root / "results" / "metrics" / f"{experiment}_history.json"
    )
    if resolved_history.is_file():
        plot_training_history(resolved_history, figure_dir, experiment)
        result["training_history_plotted"] = str(resolved_history)
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Metrics: {result_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Figures: {figure_dir}")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--history", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    evaluate(
        checkpoint=arguments.checkpoint,
        split=arguments.split,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        max_samples=arguments.max_samples,
        history_path=arguments.history,
    )


if __name__ == "__main__":
    main()
