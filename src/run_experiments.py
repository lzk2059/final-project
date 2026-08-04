"""Run all experiments for one random seed in sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


DEEP_EXPERIMENTS = (
    ("configs/resnet18.yaml", "resnet18_cross_entropy"),
    ("configs/efficientnet.yaml", "efficientnet_b0_cross_entropy"),
    (
        "configs/efficientnet_multiscale.yaml",
        "efficientnet_b0_multiscale_cross_entropy",
    ),
    ("configs/efficientnet_focal.yaml", "efficientnet_b0_cbfocal"),
    (
        "configs/efficientnet_multiscale_focal.yaml",
        "efficientnet_b0_multiscale_cbfocal",
    ),
)


def required_paths(root: Path, include_random_forest: bool) -> list[Path]:
    """Return only the files required to start the experiment sequence."""

    paths = [root / config for config, _ in DEEP_EXPERIMENTS]
    paths += [
        root / "data" / "splits" / filename
        for filename in ("train.csv", "val.csv", "test.csv")
    ]
    if include_random_forest:
        paths.append(root / "configs" / "random_forest.yaml")
        paths += [
            root / "data" / "features" / f"{split}_features.{suffix}"
            for split in ("train", "val", "test")
            for suffix in ("csv", "json")
        ]
    return paths


def deep_status(root: Path, experiment: str) -> tuple[str, Path | None]:
    """Identify whether a deep-learning run is new, done, or resumable."""

    best = root / "checkpoints" / f"{experiment}_best.pt"
    last = root / "checkpoints" / f"{experiment}_last.pt"
    history = root / "results" / "metrics" / f"{experiment}_history.json"
    if history.is_file() and best.is_file():
        result = json.loads(history.read_text(encoding="utf-8"))
        if result.get("status") == "completed":
            return "completed", None
    if last.is_file():
        return "resumable", last
    if best.exists() or history.exists():
        return "conflicting", None
    return "new", None


def random_forest_status(root: Path, experiment: str) -> str:
    """Identify whether a Random Forest run is new, done, or conflicting."""

    model = root / "checkpoints" / f"{experiment}_best.joblib"
    metrics = root / "results" / "metrics" / f"{experiment}_validation.json"
    if model.is_file() and metrics.is_file():
        return "completed"
    return "conflicting" if model.exists() or metrics.exists() else "new"


def deep_command(
    config: str, experiment: str, seed: int, debug: bool
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        config,
        "--random-seed",
        str(seed),
        "--experiment-name",
        experiment,
    ]
    if debug:
        command += [
            "--epochs", "1", "--batch-size", "4", "--no-pretrained",
            "--max-train-batches", "1", "--max-validation-batches", "1",
        ]
    return command


def evaluation_complete(root: Path, experiment: str, deep: bool) -> bool:
    """Check that metrics, predictions, and expected figures all exist."""

    evaluation_dir = root / "results" / "evaluation"
    figure_dir = root / "results" / "figures"
    evaluation_name = experiment + (
        "_validation_debug" if experiment.endswith("_debug") else "_validation"
    )
    paths = [
        evaluation_dir / f"{evaluation_name}_metrics.json",
        evaluation_dir / f"{evaluation_name}_predictions.csv",
        figure_dir / f"{evaluation_name}_confusion_matrix.png",
        figure_dir / f"{evaluation_name}_confusion_matrix_normalized.png",
        figure_dir / f"{evaluation_name}_per_class_metrics.png",
    ]
    if deep:
        paths += [
            figure_dir / f"{experiment}_{metric}.png"
            for metric in ("loss", "accuracy", "macro_f1", "learning_rate")
        ]
    return all(path.is_file() for path in paths)


def run_evaluations(
    root: Path,
    random_seed: int,
    include_random_forest: bool,
    dry_run: bool,
    debug: bool,
) -> None:
    """Evaluate every best model on validation and generate all figures."""

    models = [
        (
            f"{name}_seed{random_seed}" + ("_debug" if debug else ""),
            root / "checkpoints" / (
                f"{name}_seed{random_seed}"
                f"{'_debug' if debug else ''}_best.pt"
            ),
            True,
        )
        for _, name in DEEP_EXPERIMENTS
    ]
    if include_random_forest:
        experiment = f"random_forest_balanced_seed{random_seed}"
        if debug:
            experiment += "_debug"
        models.append(
            (
                experiment,
                root / "checkpoints" / f"{experiment}_best.joblib",
                False,
            )
        )

    print("Validation results and figures", flush=True)
    for experiment, checkpoint, deep in models:
        if evaluation_complete(root, experiment, deep):
            print(f"{experiment}: evaluation completed; skipping.")
            continue
        if not dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
        command = [
            sys.executable,
            "-m",
            "src.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "validation",
            "--batch-size",
            "4" if debug else "32",
        ]
        if debug:
            command += ["--max-samples", "24"]
        print(f"{experiment}: evaluating", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=root, check=True)


def run_experiments(
    random_seed: int,
    include_random_forest: bool = True,
    dry_run: bool = False,
    allow_cpu: bool = False,
    evaluate_results: bool = True,
    debug: bool = False,
) -> None:
    """Run one seed safely, skipping completed work and resuming interruptions."""

    if not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1.")
    root = Path(__file__).resolve().parents[1]
    missing = [str(path) for path in required_paths(root, include_random_forest) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")
    if not (allow_cpu or dry_run or debug or torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only intentionally.")

    total = len(DEEP_EXPERIMENTS) + int(include_random_forest)
    print(f"Seed {random_seed}: {total} experiments")
    for index, (config, base_name) in enumerate(DEEP_EXPERIMENTS, start=1):
        experiment = f"{base_name}_seed{random_seed}"
        output_name = experiment + ("_debug" if debug else "")
        status, resume = deep_status(root, output_name)
        print(f"[{index}/{total}] {output_name}: {status}", flush=True)
        if status == "completed":
            continue
        if status == "conflicting":
            raise RuntimeError(f"Conflicting outputs exist for {output_name}.")
        command = deep_command(config, experiment, random_seed, debug)
        if resume is not None:
            command += ["--resume", str(resume)]
        print(subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=root, check=True)

    if include_random_forest:
        experiment = f"random_forest_balanced_seed{random_seed}"
        output_name = experiment + ("_debug" if debug else "")
        status = random_forest_status(root, output_name)
        print(f"[{total}/{total}] {output_name}: {status}", flush=True)
        if status == "conflicting":
            raise RuntimeError(f"Conflicting outputs exist for {output_name}.")
        if status != "completed":
            command = [
                sys.executable,
                "-m",
                "src.train_random_forest",
                "--config",
                "configs/random_forest.yaml",
                "--random-seed",
                str(random_seed),
                "--experiment-name",
                experiment,
            ]
            if debug:
                command += [
                    "--n-estimators", "10",
                    "--max-train-samples", "120",
                    "--max-validation-samples", "120",
                ]
            print(subprocess.list2cmdline(command), flush=True)
            if not dry_run:
                subprocess.run(command, cwd=root, check=True)
    if evaluate_results:
        run_evaluations(
            root, random_seed, include_random_forest, dry_run, debug
        )
    print("Dry run complete." if dry_run else "All experiments completed.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments for one seed.")
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--skip-random-forest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a tiny end-to-end training, evaluation, and plotting test.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_experiments(
        arguments.random_seed,
        include_random_forest=not arguments.skip_random_forest,
        dry_run=arguments.dry_run,
        allow_cpu=arguments.allow_cpu,
        evaluate_results=not arguments.skip_evaluation,
        debug=arguments.debug,
    )


if __name__ == "__main__":
    main()
