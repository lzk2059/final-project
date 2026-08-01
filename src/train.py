"""Train and validate baseline infrared image classifiers."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DataLoaderConfig, create_dataloaders
from src.model import create_model, get_parameter_counts


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for one reproducible baseline experiment."""

    model_name: str = "resnet18"
    num_classes: int = 12
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    pretrained: bool = True
    random_seed: int = 42
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    history_dir: str = "results/metrics"
    optimizer_name: str = "adamw"
    loss_name: str = "cross_entropy"
    monitor: str = "validation_macro_f1"
    experiment_name: str | None = None

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive."
            )
        if self.optimizer_name.lower() != "adamw":
            raise ValueError("Only optimizer_name='adamw' is supported.")
        if self.loss_name.lower() != "cross_entropy":
            raise ValueError(
                "Only loss_name='cross_entropy' is currently supported."
            )
        if self.monitor.lower() != "validation_macro_f1":
            raise ValueError(
                "Only monitor='validation_macro_f1' is supported."
            )
        if self.experiment_name is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*",
            self.experiment_name,
        ):
            raise ValueError(
                "experiment_name may contain only letters, numbers, dots, "
                "underscores, and hyphens."
            )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use a CUDA GPU when available and otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: AdamW | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and calculate key metrics."""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    "Expected model output "
                    f"[batch_size, {num_classes}], received "
                    f"{tuple(logits.shape)}."
                )
            loss = loss_function(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(
            logits.argmax(dim=1).detach().cpu().tolist()
        )

    if total_samples == 0:
        raise RuntimeError("The data loader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(
            all_targets,
            all_predictions,
        ),
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            labels=list(range(num_classes)),
            zero_division=0,
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    validation_macro_f1: float,
    config: TrainingConfig,
) -> None:
    """Save enough state to reproduce or resume the best model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": config.model_name,
            "num_classes": config.num_classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_macro_f1": validation_macro_f1,
            "training_config": asdict(config),
        },
        path,
    )


def train_model(
    config: TrainingConfig,
    project_root: str | Path | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> dict[str, object]:
    """Train one model and retain the checkpoint with best validation Macro-F1."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    set_random_seed(config.random_seed)
    device = select_device()

    loaders = create_dataloaders(
        project_root=root,
        loader_config=DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        ),
    )
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    debug_run = (
        max_train_batches is not None
        or max_validation_batches is not None
    )
    base_experiment_name = config.experiment_name or (
        f"{config.model_name}_{config.loss_name}_seed"
        f"{config.random_seed}"
    )
    experiment_name = (
        f"{base_experiment_name}_debug"
        if debug_run
        else base_experiment_name
    )
    checkpoint_path = (
        root / config.checkpoint_dir / f"{experiment_name}_best.pt"
    )
    history_path = (
        root / config.history_dir / f"{experiment_name}_history.json"
    )

    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    print(f"Device: {device}")
    print(f"Model: {config.model_name}")
    print(f"Parameters: {get_parameter_counts(model)}")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            dataloader=loaders["train"],
            loss_function=loss_function,
            device=device,
            num_classes=config.num_classes,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        validation_metrics = run_epoch(
            model=model,
            dataloader=loaders["validation"],
            loss_function=loss_function,
            device=device,
            num_classes=config.num_classes,
            optimizer=None,
            max_batches=max_validation_batches,
        )
        epoch_seconds = time.perf_counter() - epoch_start

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_record)

        current_macro_f1 = validation_metrics["macro_f1"]
        improved = current_macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_macro_f1,
                config,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"macro-F1={train_metrics['macro_f1']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"macro-F1={current_macro_f1:.4f} | "
            f"best={best_macro_f1:.4f} | {epoch_seconds:.1f}s"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                "Early stopping: validation Macro-F1 did not improve "
                f"for {config.early_stopping_patience} epochs."
            )
            break

    total_seconds = time.perf_counter() - training_start
    result: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "parameter_counts": get_parameter_counts(model),
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "total_training_seconds": total_seconds,
        "history": history,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for a baseline training run."""

    parser = argparse.ArgumentParser(
        description="Train a baseline infrared classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment configuration file.",
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "efficientnet_b0"),
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        default=None,
        help="Do not load ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug only: limit training batches per epoch.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Debug only: limit validation batches per epoch.",
    )
    return parser.parse_args()


def load_training_config(arguments: argparse.Namespace) -> TrainingConfig:
    """Merge defaults, an optional YAML file, and explicit CLI overrides."""

    values = asdict(TrainingConfig())
    valid_fields = {field.name for field in fields(TrainingConfig)}

    if arguments.config is not None:
        config_path = arguments.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Training configuration was not found: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as file:
            yaml_values = yaml.safe_load(file)

        if yaml_values is None:
            yaml_values = {}
        if not isinstance(yaml_values, dict):
            raise TypeError(
                "The top level of the training YAML must be a mapping."
            )
        unknown_fields = set(yaml_values) - valid_fields
        if unknown_fields:
            raise ValueError(
                "Unknown training configuration field(s): "
                f"{sorted(unknown_fields)}"
            )
        values.update(yaml_values)

    command_line_overrides = {
        "model_name": arguments.model,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "early_stopping_patience": arguments.patience,
        "num_workers": arguments.num_workers,
    }
    for name, value in command_line_overrides.items():
        if value is not None:
            values[name] = value
    if arguments.no_pretrained:
        values["pretrained"] = False

    return TrainingConfig(**values)


def main() -> None:
    """Run training from command-line arguments."""

    arguments = parse_arguments()
    config = load_training_config(arguments)
    train_model(
        config,
        max_train_batches=arguments.max_train_batches,
        max_validation_batches=arguments.max_validation_batches,
    )


if __name__ == "__main__":
    main()
