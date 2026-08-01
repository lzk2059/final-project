"""Model construction utilities for 12-class infrared classification."""

from __future__ import annotations

from typing import Final

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


DEFAULT_NUMBER_OF_CLASSES: Final[int] = 12
SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "resnet18",
    "efficientnet_b0",
)


def build_resnet18(
    num_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet-18 baseline with a task-specific classifier."""

    _validate_num_classes(num_classes)
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    classifier_input_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=classifier_input_features,
        out_features=num_classes,
    )
    return model


def build_efficientnet_b0(
    num_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build an unmodified EfficientNet-B0 with a new classification head."""

    _validate_num_classes(num_classes)
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    original_classifier = model.classifier[-1]
    if not isinstance(original_classifier, nn.Linear):
        raise TypeError(
            "The final EfficientNet-B0 classifier layer is not nn.Linear."
        )

    model.classifier[-1] = nn.Linear(
        in_features=original_classifier.in_features,
        out_features=num_classes,
    )
    return model


def create_model(
    model_name: str,
    num_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create one of the supported baseline image-classification models."""

    normalized_name = model_name.strip().lower().replace("-", "_")

    builders = {
        "resnet18": build_resnet18,
        "resnet_18": build_resnet18,
        "efficientnet_b0": build_efficientnet_b0,
        "efficientnetb0": build_efficientnet_b0,
    }
    if normalized_name not in builders:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Supported models: {supported}."
        )

    return builders[normalized_name](
        num_classes=num_classes,
        pretrained=pretrained,
    )


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """Count all parameters, or only parameters that require gradients."""

    parameters = (
        parameter
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
    return sum(parameter.numel() for parameter in parameters)


def get_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for reporting."""

    return {
        "total": count_parameters(model),
        "trainable": count_parameters(model, trainable_only=True),
    }


def _validate_num_classes(num_classes: int) -> None:
    """Ensure the requested classification head has a valid size."""

    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
