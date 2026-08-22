"""Aggregate multi-seed model evaluations and create comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = {
    "random_forest_balanced": "Random Forest",
    "resnet18_cross_entropy": "ResNet-18",
    "efficientnet_b0_cross_entropy": "EfficientNet-B0",
    "efficientnet_b0_multiscale_cross_entropy": "EfficientNet + Multiscale",
    "efficientnet_b0_cbfocal": "EfficientNet + CB-Focal",
    "efficientnet_b0_multiscale_cbfocal": "Improved EfficientNet",
}
MAIN_MODELS = (
    "Random Forest",
    "ResNet-18",
    "EfficientNet-B0",
    "Improved EfficientNet",
)
ABLATION_MODELS = (
    "EfficientNet-B0",
    "EfficientNet + Multiscale",
    "EfficientNet + CB-Focal",
    "Improved EfficientNet",
)
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
CLASS_NAMES = (
    "No-Anomaly", "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
    "Hot-Spot", "Hot-Spot-Multi", "Offline-Module", "Shadowing", "Soiling",
    "Vegetation",
)


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object and reject other top-level structures."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return data


def training_seconds(root: Path, base_name: str, seed: int) -> float:
    """Load the training or model-selection duration for one run."""

    if base_name == "random_forest_balanced":
        path = root / "results" / "metrics" / f"{base_name}_seed{seed}_validation.json"
        return float(read_json(path)["total_selection_seconds"])
    path = root / "results" / "metrics" / f"{base_name}_seed{seed}_history.json"
    return float(read_json(path)["total_training_seconds"])


def load_runs(root: Path, split: str, seeds: list[int]) -> pd.DataFrame:
    """Load and validate one evaluation result for every model and seed."""

    rows = []
    expected_samples: int | None = None
    for base_name, display_name in EXPERIMENTS.items():
        for seed in seeds:
            experiment = f"{base_name}_seed{seed}"
            path = root / "results" / "evaluation" / f"{experiment}_{split}_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation result: {path}")
            result = read_json(path)
            if result.get("split") != split:
                raise ValueError(f"Incorrect split recorded in {path}.")
            metrics = result.get("metrics")
            details = result.get("model_details")
            if not isinstance(metrics, dict) or not isinstance(details, dict):
                raise ValueError(f"Missing metrics or model_details in {path}.")
            samples = int(metrics["number_of_samples"])
            expected_samples = samples if expected_samples is None else expected_samples
            if samples != expected_samples:
                raise ValueError(f"Inconsistent sample count in {path}: {samples}")
            values = [float(metrics[name]) for name in METRICS]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite metric found in {path}.")
            counts = details.get("parameter_counts")
            rows.append(
                {
                    "model": display_name,
                    "experiment": experiment,
                    "seed": seed,
                    "samples": samples,
                    "accuracy": values[0],
                    "macro_f1": values[1],
                    "balanced_accuracy": values[2],
                    "parameters": counts.get("total") if isinstance(counts, dict) else np.nan,
                    "training_seconds": training_seconds(root, base_name, seed),
                    "inference_seconds": float(details["inference_seconds"]),
                    "inference_ms_per_image": 1000 * float(details["inference_seconds_per_image"]),
                }
            )
    return pd.DataFrame(rows)


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and sample standard deviation across seeds."""

    columns = [
        *METRICS,
        "parameters",
        "training_seconds",
        "inference_seconds",
        "inference_ms_per_image",
    ]
    grouped = runs.groupby("model", sort=False)[columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def summarise_per_class(root: Path, split: str, seeds: list[int]) -> pd.DataFrame:
    """Aggregate per-class Precision, Recall, and F1 across seeds."""

    rows = []
    for base_name, display_name in EXPERIMENTS.items():
        for seed in seeds:
            path = root / "results" / "evaluation" / f"{base_name}_seed{seed}_{split}_metrics.json"
            per_class = read_json(path)["metrics"]["per_class"]
            for class_name, values in per_class.items():
                rows.append(
                    {
                        "model": display_name,
                        "class_name": class_name,
                        "seed": seed,
                        "precision": float(values["precision"]),
                        "recall": float(values["recall"]),
                        "f1": float(values["f1"]),
                        "support": int(values["support"]),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "class_name"], sort=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support=("support", "first"),
    )
    return summary.reset_index()


def select_models(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """Select models and preserve the intended presentation order."""

    indexed = frame.set_index("model")
    missing = [name for name in names if name not in indexed.index]
    if missing:
        raise ValueError(f"Missing models from summary: {missing}")
    return indexed.loc[list(names)].reset_index()


def plot_metric(frame: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """Save one readable mean-with-standard-deviation comparison plot."""

    means = frame[f"{metric}_mean"].to_numpy(dtype=float)
    errors = frame[f"{metric}_std"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(positions, means, yerr=errors, capsize=5, color="#4472C4")
    axis.set_xticks(positions, frame["model"], rotation=18, ha="right")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title)
    axis.set_ylim(0, min(1.0, max(means + errors) + 0.12))
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_improved_per_class_f1(per_class: pd.DataFrame, path: Path) -> None:
    """Plot mean per-class F1 with standard deviation across random seeds."""

    frame = per_class[per_class["model"] == "Improved EfficientNet"].set_index(
        "class_name"
    )
    missing = [name for name in CLASS_NAMES if name not in frame.index]
    if missing:
        raise ValueError(f"Missing Improved EfficientNet classes: {missing}")
    frame = frame.loc[list(CLASS_NAMES)]
    means = frame["f1_mean"].to_numpy(dtype=float)
    errors = frame["f1_std"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(positions, means, yerr=errors, capsize=4, color="#4472C4")
    axis.set_xticks(positions, CLASS_NAMES, rotation=35, ha="right")
    axis.set_ylabel("F1-score")
    axis.set_title("Improved EfficientNet Per-Class Test F1 (Mean ± SD)")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def compare_results(root: Path, split: str, seeds: list[int]) -> None:
    """Create reproducible tables and figures from completed evaluations."""

    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct random seeds.")
    output_dir = root / "results" / "comparison"
    figure_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root, split, seeds)
    summary = summarise_runs(runs)
    main = select_models(summary, MAIN_MODELS)
    ablation = select_models(summary, ABLATION_MODELS)
    per_class = summarise_per_class(root, split, seeds)

    runs.to_csv(output_dir / f"{split}_all_runs.csv", index=False)
    summary.to_csv(output_dir / f"{split}_all_models_summary.csv", index=False)
    main.to_csv(output_dir / f"{split}_main_models.csv", index=False)
    ablation.to_csv(output_dir / f"{split}_ablation.csv", index=False)
    per_class.to_csv(output_dir / f"{split}_per_class_summary.csv", index=False)
    payload = {
        "split": split,
        "seeds": seeds,
        "number_of_runs": len(runs),
        "summary": summary.astype(object).where(pd.notna(summary), None).to_dict(
            orient="records"
        ),
    }
    (output_dir / f"{split}_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for metric in METRICS:
        plot_metric(
            main,
            metric,
            f"Main Model Comparison ({split.title()})",
            figure_dir / f"comparison_{split}_main_{metric}.png",
        )
        plot_metric(
            ablation,
            metric,
            f"EfficientNet Ablation ({split.title()})",
            figure_dir / f"comparison_{split}_ablation_{metric}.png",
        )

    if split == "test":
        plot_improved_per_class_f1(
            per_class,
            figure_dir / "improved_efficientnet_test_per_class_f1_mean_std.png",
        )

    winner = summary.loc[summary["macro_f1_mean"].idxmax()]
    print(f"Runs: {len(runs)} | Seeds: {seeds} | Split: {split}")
    print(f"Best mean Macro-F1: {winner['model']} = {winner['macro_f1_mean']:.4f} ± {winner['macro_f1_std']:.4f}")
    print(f"Tables: {output_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed multi-seed experiments.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    compare_results(Path(__file__).resolve().parents[1], arguments.split, arguments.seeds)


if __name__ == "__main__":
    main()
