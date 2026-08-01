"""Loss functions for imbalanced infrared image classification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def load_class_counts(
    csv_path: str | Path,
    num_classes: int,
    label_column: str = "class_index",
) -> torch.Tensor:
    """Count each class using a training-split CSV only."""

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training split CSV was not found: {path}")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    dataframe = pd.read_csv(path, usecols=lambda name: name == label_column)
    if label_column not in dataframe.columns:
        raise ValueError(
            f"{path.name} does not contain column '{label_column}'."
        )

    labels = pd.to_numeric(dataframe[label_column], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path.name} contains missing or invalid labels.")
    if not (labels == labels.astype(int)).all():
        raise ValueError(f"{path.name} contains non-integer labels.")
    labels = labels.astype(int)
    if not labels.between(0, num_classes - 1).all():
        raise ValueError(
            f"{path.name} contains labels outside 0-{num_classes - 1}."
        )

    counts = labels.value_counts().reindex(
        range(num_classes),
        fill_value=0,
    )
    if (counts == 0).any():
        missing_classes = counts[counts == 0].index.tolist()
        raise ValueError(
            "Every class must occur in the training split. Missing class "
            f"indices: {missing_classes}"
        )

    return torch.tensor(counts.to_numpy(), dtype=torch.long)


def calculate_class_balanced_weights(
    class_counts: Sequence[int] | torch.Tensor,
    beta: float = 0.9999,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate normalized effective-number weights for every class."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in the interval [0, 1).")

    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() <= 1:
        raise ValueError("class_counts must be a one-dimensional sequence.")
    if not torch.isfinite(counts).all():
        raise ValueError("class_counts must contain only finite values.")
    if (counts <= 0).any():
        raise ValueError("Every class count must be greater than zero.")
    if not torch.equal(counts, counts.round()):
        raise ValueError("Every class count must be an integer.")

    if beta == 0.0:
        weights = torch.ones_like(counts)
    else:
        # -expm1(n * log(beta)) computes 1 - beta**n accurately when beta is
        # very close to one, which is the normal class-balanced-loss setting.
        effective_numbers = -torch.expm1(counts * math.log(beta))
        weights = (1.0 - beta) / effective_numbers

    # A mean weight of one keeps the overall loss scale comparable across
    # different beta values and class distributions.
    weights = weights / weights.mean()
    return weights.to(dtype=dtype)


class ClassBalancedFocalLoss(nn.Module):
    """Multi-class focal loss weighted by effective sample numbers."""

    def __init__(
        self,
        class_counts: Sequence[int] | torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma cannot be negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'."
            )

        weights = calculate_class_balanced_weights(
            class_counts=class_counts,
            beta=beta,
        )
        self.register_buffer("class_weights", weights)
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = weights.numel()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss from unnormalized logits and integer targets."""

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [batch_size, num_classes]."
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} output classes, received "
                f"{logits.shape[1]}."
            )
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                "targets must have shape [batch_size] matching logits."
            )
        if targets.dtype != torch.long:
            raise TypeError("targets must use torch.long integer labels.")
        if targets.numel() == 0:
            raise ValueError("The batch cannot be empty.")
        if targets.min().item() < 0 or targets.max().item() >= self.num_classes:
            raise ValueError(
                f"targets must be between 0 and {self.num_classes - 1}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN or infinite values.")

        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        sample_weights = self.class_weights[targets]
        focal_factor = (1.0 - target_probabilities).pow(self.gamma)
        losses = (
            -sample_weights
            * focal_factor
            * target_log_probabilities
        )

        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()

        # Match torch.nn.CrossEntropyLoss(weight=..., reduction="mean") when
        # gamma is zero by dividing by the selected class-weight sum.
        return losses.sum() / sample_weights.sum()


def build_class_balanced_focal_loss_from_csv(
    training_csv_path: str | Path,
    num_classes: int,
    beta: float = 0.9999,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> ClassBalancedFocalLoss:
    """Build the loss using class counts from the training split."""

    class_counts = load_class_counts(
        csv_path=training_csv_path,
        num_classes=num_classes,
    )
    return ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=beta,
        gamma=gamma,
        reduction=reduction,
    )
