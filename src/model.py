"""Model construction utilities for 12-class infrared classification."""

from __future__ import annotations

from typing import Final

import torch
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
    "efficientnet_b0_multiscale",
)

MULTISCALE_FEATURE_INDICES: Final[tuple[int, int, int]] = (3, 5, 8)
MULTISCALE_INPUT_CHANNELS: Final[tuple[int, int, int]] = (40, 112, 1280)


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


class MultiScaleEfficientNetB0(nn.Module):
    """EfficientNet-B0 with pooled feature fusion from three depths."""

    def __init__(
        self,
        num_classes: int = DEFAULT_NUMBER_OF_CLASSES,
        pretrained: bool = True,
        projection_channels: int = 128,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_num_classes(num_classes)
        if projection_channels <= 0:
            raise ValueError("projection_channels must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "dropout_probability must be in the interval [0, 1)."
            )

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.feature_indices = MULTISCALE_FEATURE_INDICES
        self.feature_channels = MULTISCALE_INPUT_CHANNELS
        self.projection_channels = projection_channels

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        projection_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(projection_channels),
                    nn.SiLU(inplace=True),
                    nn.AdaptiveAvgPool2d(output_size=1),
                )
                for input_channels in self.feature_channels
            ]
        )
        fused_channels = projection_channels * len(self.feature_indices)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_probability, inplace=True),
            nn.Linear(fused_channels, num_classes),
        )

    def extract_multiscale_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return feature maps from the configured backbone stages."""

        selected_features: list[torch.Tensor] = []
        x = images
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.feature_indices:
                selected_features.append(x)

        if len(selected_features) != len(self.feature_indices):
            raise RuntimeError(
                "EfficientNet-B0 did not produce every selected feature map."
            )
        return tuple(selected_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Fuse pooled features and return unnormalized class logits."""

        feature_maps = self.extract_multiscale_features(images)
        projected_vectors = [
            projection(feature_map).flatten(start_dim=1)
            for projection, feature_map in zip(
                self.projections,
                feature_maps,
                strict=True,
            )
        ]
        fused_features = torch.cat(projected_vectors, dim=1)
        return self.classifier(fused_features)


def build_efficientnet_b0_multiscale(
    num_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build the multi-scale EfficientNet-B0 experiment model."""

    return MultiScaleEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
    )


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
        "efficientnet_b0_multiscale": build_efficientnet_b0_multiscale,
        "efficientnetb0_multiscale": build_efficientnet_b0_multiscale,
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
