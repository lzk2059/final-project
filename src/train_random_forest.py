"""Train and validate the balanced Random Forest baseline."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from src.features import METADATA_COLUMNS


@dataclass(frozen=True)
class RandomForestConfig:
    """Reproducible Random Forest experiment settings."""

    experiment_name: str = "random_forest_balanced_seed42"
    num_classes: int = 12
    random_seed: int = 42
    class_weight: str = "balanced"
    n_jobs: int = -1
    n_estimators: tuple[int, ...] = (300,)
    max_depth: tuple[int | None, ...] = (None, 20)
    min_samples_leaf: tuple[int, ...] = (1, 2)
    max_features: tuple[str | float | None, ...] = ("sqrt",)
    feature_dir: str = "data/features"
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "results/metrics"

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed must be between 0 and 2**32 - 1.")
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be 'balanced'.")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", self.experiment_name
        ):
            raise ValueError("experiment_name contains invalid characters.")
        if not self.n_estimators or min(self.n_estimators) <= 0:
            raise ValueError("n_estimators values must be positive.")
        if any(value is not None and value <= 0 for value in self.max_depth):
            raise ValueError("max_depth values must be positive or null.")
        if not self.min_samples_leaf or min(self.min_samples_leaf) <= 0:
            raise ValueError("min_samples_leaf values must be positive.")
        if not self.max_features:
            raise ValueError("At least one max_features value is required.")


def load_config(path: str | Path) -> RandomForestConfig:
    """Load a YAML configuration and reject unknown fields."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration was not found: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Random Forest YAML must contain a mapping.")

    valid_fields = {field.name for field in fields(RandomForestConfig)}
    unknown = set(values) - valid_fields
    if unknown:
        raise ValueError(f"Unknown configuration field(s): {sorted(unknown)}")
    for name in (
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    ):
        if name in values:
            value = values[name]
            values[name] = tuple(value if isinstance(value, list) else [value])
    return RandomForestConfig(**values)


def load_feature_split(
    feature_dir: Path,
    split_name: str,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one feature split and perform essential integrity checks."""

    csv_path = feature_dir / f"{split_name}_features.csv"
    json_path = feature_dir / f"{split_name}_features.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Missing feature files for split: {split_name}")

    dataframe = pd.read_csv(csv_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Invalid feature schema: {json_path}")
    if dataframe.columns.tolist() != list(METADATA_COLUMNS) + feature_columns:
        raise ValueError(f"Feature columns do not match: {csv_path.name}")
    if metadata.get("number_of_samples") != len(dataframe):
        raise ValueError(f"Sample count does not match: {csv_path.name}")

    x = dataframe[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values found in {csv_path.name}.")

    y = validate_class_labels(
        dataframe["class_index"], num_classes, csv_path.name
    )
    if set(np.unique(y)) != set(range(num_classes)):
        raise ValueError(f"{csv_path.name} must contain classes 0-{num_classes - 1}.")
    return x, y, feature_columns


def validate_class_labels(
    labels: pd.Series,
    num_classes: int,
    source_name: str,
) -> np.ndarray:
    """Reject non-numeric, fractional, non-finite, and out-of-range labels."""

    labels = pd.to_numeric(labels, errors="coerce")
    label_values = labels.to_numpy(dtype=np.float64)
    if not np.isfinite(label_values).all():
        raise ValueError(f"Invalid labels found in {source_name}.")
    if not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError(f"Non-integer labels found in {source_name}.")
    if not ((0 <= label_values) & (label_values < num_classes)).all():
        raise ValueError(
            f"Labels in {source_name} must be between 0 and "
            f"{num_classes - 1}."
        )
    return label_values.astype(np.int64)


def stratified_subset(
    x: np.ndarray,
    y: np.ndarray,
    size: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible debug subset while retaining every class."""

    if size is None or size >= len(y):
        return x, y
    if size < len(np.unique(y)):
        raise ValueError("Debug sample count must cover every class.")
    x_subset, _, y_subset, _ = train_test_split(
        x,
        y,
        train_size=size,
        stratify=y,
        random_state=random_seed,
    )
    return x_subset, y_subset


def order_class_names(
    mapping: dict[str, object],
    num_classes: int,
) -> list[str]:
    """Validate a label mapping and return names in numeric order."""

    names = [""] * num_classes
    used_indices: set[int] = set()
    for name, raw_index in mapping.items():
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, float))
            or not float(raw_index).is_integer()
        ):
            raise ValueError(f"Class '{name}' has a non-integer index.")
        index = int(raw_index)
        if not 0 <= index < num_classes:
            raise ValueError(
                f"Class '{name}' has index {index}, outside "
                f"0-{num_classes - 1}."
            )
        if index in used_indices:
            raise ValueError(f"Class index {index} is assigned more than once.")
        used_indices.add(index)
        names[index] = str(name)

    expected_indices = set(range(num_classes))
    if used_indices != expected_indices:
        missing = sorted(expected_indices - used_indices)
        raise ValueError(f"Label mapping is missing class indices: {missing}")
    return names


def load_class_names(root: Path, num_classes: int) -> list[str]:
    """Load and validate class names in their fixed numeric order."""

    path = root / "data" / "processed" / "label_to_index.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise TypeError("label_to_index.json must contain a mapping.")
    return order_class_names(mapping, num_classes)


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Calculate validation metrics in a fixed class order."""

    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
    }


def train_random_forest(
    config: RandomForestConfig,
    project_root: str | Path | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> dict[str, object]:
    """Select and save the model with highest validation Macro-F1."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    feature_dir = root / config.feature_dir
    x_train, y_train, feature_columns = load_feature_split(
        feature_dir, "train", config.num_classes
    )
    x_val, y_val, val_columns = load_feature_split(
        feature_dir, "val", config.num_classes
    )
    if feature_columns != val_columns:
        raise ValueError("Training and validation feature schemas differ.")

    x_train, y_train = stratified_subset(
        x_train, y_train, max_train_samples, config.random_seed
    )
    x_val, y_val = stratified_subset(
        x_val, y_val, max_validation_samples, config.random_seed
    )
    class_names = load_class_names(root, config.num_classes)
    debug = max_train_samples is not None or max_validation_samples is not None
    experiment = config.experiment_name + ("_debug" if debug else "")
    parameter_grid = ParameterGrid(
        {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "max_features": config.max_features,
        }
    )

    best: tuple[tuple[float, float, float], RandomForestClassifier, dict, dict] | None = None
    trials: list[dict[str, object]] = []
    selection_start = time.perf_counter()

    for index, parameters in enumerate(parameter_grid, start=1):
        model = RandomForestClassifier(
            **parameters,
            class_weight=config.class_weight,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
        )
        start = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start
        metrics = calculate_metrics(y_val, model.predict(x_val), class_names)
        score = (
            float(metrics["macro_f1"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["accuracy"]),
        )
        trials.append(
            {
                "parameters": parameters,
                "training_seconds": training_seconds,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        print(
            f"Trial {index} | {parameters} | "
            f"Macro-F1={score[0]:.4f} | Balanced Accuracy={score[1]:.4f}"
        )
        if best is None or score > best[0]:
            best = (score, model, parameters, metrics)

    if best is None:
        raise RuntimeError("No Random Forest candidate was trained.")
    _, best_model, best_parameters, best_metrics = best
    result = {
        "config": asdict(config),
        "experiment_name": experiment,
        "debug_run": debug,
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "number_of_features": len(feature_columns),
        "selection_metric": "validation_macro_f1",
        "best_parameters": best_parameters,
        "best_validation_metrics": best_metrics,
        "total_selection_seconds": time.perf_counter() - selection_start,
        "trials": trials,
    }

    checkpoint = root / config.checkpoint_dir / f"{experiment}_best.joblib"
    metrics_path = root / config.metrics_dir / f"{experiment}_validation.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "class_names": class_names,
            "config": asdict(config),
            "best_parameters": best_parameters,
            "best_validation_metrics": best_metrics,
        },
        checkpoint,
    )
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best model: {checkpoint}")
    print(f"Validation results: {metrics_path}")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a balanced Random Forest baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/random_forest.yaml"),
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    values = asdict(load_config(arguments.config))
    overrides = {
        "random_seed": arguments.random_seed,
        "experiment_name": arguments.experiment_name,
        "n_estimators": (
            (arguments.n_estimators,)
            if arguments.n_estimators is not None
            else None
        ),
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    train_random_forest(
        RandomForestConfig(**values),
        max_train_samples=arguments.max_train_samples,
        max_validation_samples=arguments.max_validation_samples,
    )


if __name__ == "__main__":
    main()
