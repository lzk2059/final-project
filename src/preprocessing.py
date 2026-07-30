"""Image preprocessing and training-only augmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


# EfficientNet-B0 and ResNet-18 will use ImageNet-pretrained weights. Grayscale
# infrared images are therefore copied to three channels and normalized with
# the statistics expected by those weights.
IMAGENET_MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)
IMAGENET_STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration shared by training, validation, test, and inference."""

    image_size: tuple[int, int] = (224, 224)
    # None uses the median intensity of each image. This avoids introducing
    # large artificial black borders when a 24 x 40 image is letterboxed.
    padding_value: int | None = None

    # Conservative augmentation is used because the source images are only
    # 24 x 40 pixels. Values here are intentionally small.
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 7.0
    translation_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.95, 1.05)
    noise_probability: float = 0.20
    noise_standard_deviation: float = 0.01

    def __post_init__(self) -> None:
        """Reject invalid settings before a training run starts."""

        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size values must be positive.")
        if (
            self.padding_value is not None
            and not 0 <= self.padding_value <= 255
        ):
            raise ValueError(
                "padding_value must be None or between 0 and 255."
            )
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError(
                "horizontal_flip_probability must be between 0 and 1."
            )
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees cannot be negative.")
        if not 0 <= self.translation_fraction <= 1:
            raise ValueError(
                "translation_fraction must be between 0 and 1."
            )
        minimum_scale, maximum_scale = self.scale_range
        if minimum_scale <= 0 or maximum_scale < minimum_scale:
            raise ValueError("scale_range must contain valid positive values.")
        if not 0 <= self.noise_probability <= 1:
            raise ValueError("noise_probability must be between 0 and 1.")
        if self.noise_standard_deviation < 0:
            raise ValueError(
                "noise_standard_deviation cannot be negative."
            )


class ResizeWithPadding:
    """Resize a PIL image without distortion and pad it to a fixed size."""

    def __init__(
        self,
        output_size: tuple[int, int],
        fill: int | None = None,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        self.output_height, self.output_width = output_size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a letterboxed image with exactly ``output_size``."""

        if not isinstance(image, Image.Image):
            raise TypeError("ResizeWithPadding expects a PIL image.")

        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("The input image has an invalid size.")

        scale = min(
            self.output_width / source_width,
            self.output_height / source_height,
        )
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))

        resized = image.resize(
            (resized_width, resized_height),
            resample=self.interpolation,
        )

        horizontal_padding = self.output_width - resized_width
        vertical_padding = self.output_height - resized_height
        left = horizontal_padding // 2
        right = horizontal_padding - left
        top = vertical_padding // 2
        bottom = vertical_padding - top

        # A per-image median gives the padded region a neutral infrared
        # intensity. An explicit integer can still be supplied for controlled
        # experiments that need a fixed padding value.
        padding_fill = self.fill
        if padding_fill is None:
            padding_fill = round(
                ImageStat.Stat(image.convert("L")).median[0]
            )

        return ImageOps.expand(
            resized,
            border=(left, top, right, bottom),
            fill=padding_fill,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_size=({self.output_height}, {self.output_width}), "
            f"fill={self.fill})"
        )


class RandomGaussianNoise:
    """Add low-amplitude Gaussian noise to a tensor with a probability."""

    def __init__(
        self,
        probability: float = 0.20,
        standard_deviation: float = 0.01,
    ) -> None:
        self.probability = probability
        self.standard_deviation = standard_deviation

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Add noise before normalization and keep pixels in [0, 1]."""

        if torch.rand(1).item() >= self.probability:
            return image

        if image.ndim != 3:
            raise ValueError(
                "RandomGaussianNoise expects a [channels, height, width] "
                "tensor."
            )

        # The source infrared image is grayscale and has merely been copied to
        # three channels. Reuse one noise map across all channels so the
        # augmentation does not invent artificial colour differences.
        shared_noise = torch.randn(
            (1, image.shape[-2], image.shape[-1]),
            dtype=image.dtype,
            device=image.device,
        )
        noise = shared_noise * self.standard_deviation
        return torch.clamp(image + noise, min=0.0, max=1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"probability={self.probability}, "
            f"standard_deviation={self.standard_deviation})"
        )


class RandomAffineWithMedianFill:
    """Apply a small random affine transform using a neutral image fill."""

    def __init__(
        self,
        degrees: float,
        translation_fraction: float,
        scale_range: tuple[float, float],
        fixed_fill: int | None = None,
    ) -> None:
        self.degrees = (-degrees, degrees)
        self.translate = (
            translation_fraction,
            translation_fraction,
        )
        self.scale_range = scale_range
        self.fixed_fill = fixed_fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Sample affine parameters and fill exposed pixels neutrally."""

        if not isinstance(image, Image.Image):
            raise TypeError(
                "RandomAffineWithMedianFill expects a PIL image."
            )

        angle, translations, scale, shear = (
            transforms.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale_range,
                shears=None,
                img_size=list(image.size),
            )
        )
        fill = self.fixed_fill
        if fill is None:
            fill = round(
                ImageStat.Stat(image.convert("L")).median[0]
            )

        return transform_functional.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )


def build_preprocessing(
    training: bool,
    config: PreprocessingConfig | None = None,
) -> transforms.Compose:
    """
    Build the preprocessing pipeline for one dataset split.

    Random augmentation is included only when ``training`` is true.
    Validation, test, and single-image inference should all call this function
    with ``training=False`` so that they use identical deterministic steps.
    """

    config = config or PreprocessingConfig()

    common_start: list[object] = [
        # Force a predictable source mode even if a future image is RGB.
        transforms.Grayscale(num_output_channels=1),
        ResizeWithPadding(
            output_size=config.image_size,
            fill=config.padding_value,
        ),
    ]

    training_only: list[object] = []
    if training:
        training_only = [
            transforms.RandomHorizontalFlip(
                p=config.horizontal_flip_probability
            ),
            RandomAffineWithMedianFill(
                degrees=config.rotation_degrees,
                translation_fraction=config.translation_fraction,
                scale_range=config.scale_range,
                fixed_fill=config.padding_value,
            ),
        ]

    common_end: list[object] = [
        # Replicate the grayscale signal into three equal channels.
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]

    if training:
        common_end.append(
            RandomGaussianNoise(
                probability=config.noise_probability,
                standard_deviation=config.noise_standard_deviation,
            )
        )

    common_end.append(
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
    )

    return transforms.Compose(
        common_start + training_only + common_end
    )


def build_train_preprocessing(
    config: PreprocessingConfig | None = None,
) -> transforms.Compose:
    """Return preprocessing with training-only random augmentation."""

    return build_preprocessing(training=True, config=config)


def build_evaluation_preprocessing(
    config: PreprocessingConfig | None = None,
) -> transforms.Compose:
    """Return deterministic preprocessing for validation, test, and inference."""

    return build_preprocessing(training=False, config=config)
