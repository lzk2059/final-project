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


def deep_command(config: str, experiment: str, seed: int) -> list[str]:
    return [
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


def run_experiments(
    random_seed: int,
    include_random_forest: bool = True,
    dry_run: bool = False,
    allow_cpu: bool = False,
) -> None:
    """Run one seed safely, skipping completed work and resuming interruptions."""

    if not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1.")
    root = Path(__file__).resolve().parents[1]
    missing = [str(path) for path in required_paths(root, include_random_forest) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")
    if not (allow_cpu or dry_run or torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only intentionally.")

    total = len(DEEP_EXPERIMENTS) + int(include_random_forest)
    print(f"Seed {random_seed}: {total} experiments")
    for index, (config, base_name) in enumerate(DEEP_EXPERIMENTS, start=1):
        experiment = f"{base_name}_seed{random_seed}"
        status, resume = deep_status(root, experiment)
        print(f"[{index}/{total}] {experiment}: {status}", flush=True)
        if status == "completed":
            continue
        if status == "conflicting":
            raise RuntimeError(f"Conflicting outputs exist for {experiment}.")
        command = deep_command(config, experiment, random_seed)
        if resume is not None:
            command += ["--resume", str(resume)]
        print(subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=root, check=True)

    if include_random_forest:
        experiment = f"random_forest_balanced_seed{random_seed}"
        status = random_forest_status(root, experiment)
        print(f"[{total}/{total}] {experiment}: {status}", flush=True)
        if status == "conflicting":
            raise RuntimeError(f"Conflicting outputs exist for {experiment}.")
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
            print(subprocess.list2cmdline(command), flush=True)
            if not dry_run:
                subprocess.run(command, cwd=root, check=True)
    print("Dry run complete." if dry_run else "All experiments completed.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments for one seed.")
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--skip-random-forest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_experiments(
        arguments.random_seed,
        include_random_forest=not arguments.skip_random_forest,
        dry_run=arguments.dry_run,
        allow_cpu=arguments.allow_cpu,
    )


if __name__ == "__main__":
    main()
