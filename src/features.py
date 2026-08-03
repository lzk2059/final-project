"""Extract handcrafted intensity, LBP, and GLCM image features."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


INTENSITY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "intensity_median",
    "intensity_q25",
    "intensity_q75",
    "intensity_skewness",
    "intensity_kurtosis",
)
GLCM_PROPERTIES: Final[tuple[str, ...]] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)
SPLIT_FILENAMES: Final[dict[str, str]] = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}
FEATURE_SCHEMA_VERSION: Final[int] = 1
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "image_id",
    "image_path",
    "class_index",
    "class_label",
)


@dataclass(frozen=True)
class HandcraftedFeatureConfig:
    """Fixed parameters for reproducible handcrafted feature extraction."""

    lbp_radius: int = 1
    lbp_points: int = 8
    lbp_method: str = "uniform"
    glcm_levels: int = 16
    glcm_distances: tuple[int, ...] = (1, 2)
    glcm_angles: tuple[float, ...] = (
        0.0,
        math.pi / 4,
        math.pi / 2,
        3 * math.pi / 4,
    )

    def __post_init__(self) -> None:
        if self.lbp_radius <= 0:
            raise ValueError("lbp_radius must be positive.")
        if self.lbp_points <= 0:
            raise ValueError("lbp_points must be positive.")
        if self.lbp_method != "uniform":
            raise ValueError("Only lbp_method='uniform' is supported.")
        if not 2 <= self.glcm_levels <= 256:
            raise ValueError("glcm_levels must be between 2 and 256.")
        if not self.glcm_distances or any(
            distance <= 0 for distance in self.glcm_distances
        ):
            raise ValueError("Every GLCM distance must be positive.")
        if not self.glcm_angles:
            raise ValueError("At least one GLCM angle is required.")

    @property
    def lbp_bins(self) -> int:
        """Return the histogram size for uniform LBP."""

        return self.lbp_points + 2


def extract_intensity_features(image: np.ndarray) -> dict[str, float]:
    """Calculate normalized pixel-intensity summary statistics."""

    pixels = image.astype(np.float64).reshape(-1) / 255.0
    mean = float(pixels.mean())
    standard_deviation = float(pixels.std(ddof=0))

    if standard_deviation > 0:
        standardized = (pixels - mean) / standard_deviation
        skewness = float(np.mean(standardized**3))
        # Excess kurtosis makes a Gaussian distribution equal to zero.
        kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "intensity_mean": mean,
        "intensity_std": standard_deviation,
        "intensity_min": float(pixels.min()),
        "intensity_max": float(pixels.max()),
        "intensity_median": float(np.median(pixels)),
        "intensity_q25": float(np.quantile(pixels, 0.25)),
        "intensity_q75": float(np.quantile(pixels, 0.75)),
        "intensity_skewness": skewness,
        "intensity_kurtosis": kurtosis,
    }


def extract_lbp_features(
    image: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> dict[str, float]:
    """Calculate a normalized uniform-LBP histogram."""

    lbp = local_binary_pattern(
        image,
        P=config.lbp_points,
        R=config.lbp_radius,
        method=config.lbp_method,
    )
    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(config.lbp_bins + 1),
        range=(0, config.lbp_bins),
    )
    histogram = histogram.astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    return {
        f"lbp_bin_{index:02d}": float(value)
        for index, value in enumerate(histogram)
    }


def extract_glcm_features(
    image: np.ndarray,
    config: HandcraftedFeatureConfig,
) -> dict[str, float]:
    """Calculate mean and standard deviation of GLCM properties."""

    quantized = np.floor(
        image.astype(np.float64) * config.glcm_levels / 256.0
    ).astype(np.uint8)
    quantized = np.clip(quantized, 0, config.glcm_levels - 1)

    matrix = graycomatrix(
        quantized,
        distances=config.glcm_distances,
        angles=config.glcm_angles,
        levels=config.glcm_levels,
        symmetric=True,
        normed=True,
    )
    features: dict[str, float] = {}
    for property_name in GLCM_PROPERTIES:
        values = graycoprops(matrix, property_name)
        output_name = property_name.lower()
        features[f"glcm_{output_name}_mean"] = float(values.mean())
        features[f"glcm_{output_name}_std"] = float(values.std(ddof=0))
    return features


def extract_handcrafted_features(
    image: np.ndarray,
    config: HandcraftedFeatureConfig | None = None,
) -> dict[str, float]:
    """Extract all handcrafted features from one uint8 grayscale image."""

    config = config or HandcraftedFeatureConfig()
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array.")
    if image.size == 0:
        raise ValueError("image cannot be empty.")
    if image.dtype != np.uint8:
        raise TypeError("image must use np.uint8 pixel values.")

    features = extract_intensity_features(image)
    features.update(extract_lbp_features(image, config))
    features.update(extract_glcm_features(image, config))

    values = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Extracted features contain NaN or infinite values.")
    return features


def get_feature_names(
    config: HandcraftedFeatureConfig | None = None,
) -> list[str]:
    """Return the deterministic ordered handcrafted-feature schema."""

    config = config or HandcraftedFeatureConfig()
    names = list(INTENSITY_FEATURE_NAMES)
    names.extend(
        f"lbp_bin_{index:02d}"
        for index in range(config.lbp_bins)
    )
    for property_name in GLCM_PROPERTIES:
        output_name = property_name.lower()
        names.extend(
            (
                f"glcm_{output_name}_mean",
                f"glcm_{output_name}_std",
            )
        )
    return names


def save_feature_metadata(
    feature_dataframe: pd.DataFrame,
    split_csv_path: str | Path,
    feature_csv_path: str | Path,
    config: HandcraftedFeatureConfig | None = None,
    max_samples: int | None = None,
) -> Path:
    """Atomically save the metadata needed to reproduce a feature CSV."""

    split_path = Path(split_csv_path).expanduser().resolve()
    feature_path = Path(feature_csv_path).expanduser().resolve()
    config = config or HandcraftedFeatureConfig()

    if not split_path.is_file():
        raise FileNotFoundError(f"Split CSV was not found: {split_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature CSV was not found: {feature_path}")

    expected_feature_names = get_feature_names(config)
    expected_columns = list(METADATA_COLUMNS) + expected_feature_names
    if feature_dataframe.columns.tolist() != expected_columns:
        raise ValueError(
            "Feature CSV columns do not match the current feature schema."
        )

    class_counts = (
        feature_dataframe["class_index"]
        .value_counts()
        .sort_index()
    )
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_split": split_path.name,
        "debug_extraction": max_samples is not None,
        "max_samples": max_samples,
        "number_of_samples": int(len(feature_dataframe)),
        "number_of_metadata_columns": len(METADATA_COLUMNS),
        "number_of_features": len(expected_feature_names),
        "metadata_columns": list(METADATA_COLUMNS),
        "feature_columns": expected_feature_names,
        "class_counts": {
            str(int(index)): int(count)
            for index, count in class_counts.items()
        },
        "image_source": "original_grayscale_no_resize",
        "intensity_features": list(INTENSITY_FEATURE_NAMES),
        "lbp": {
            "radius": config.lbp_radius,
            "points": config.lbp_points,
            "method": config.lbp_method,
            "bins": config.lbp_bins,
            "histogram_normalized": True,
        },
        "glcm": {
            "levels": config.glcm_levels,
            "distances": list(config.glcm_distances),
            "angles_degrees": [
                float(math.degrees(angle))
                for angle in config.glcm_angles
            ],
            "symmetric": True,
            "normalized": True,
            "properties": list(GLCM_PROPERTIES),
        },
        "feature_validation_performed": True,
    }

    metadata_path = feature_path.with_suffix(".json")
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    print(f"Saved feature metadata: {metadata_path}")
    return metadata_path


def _resolve_image_path(project_root: Path, value: object) -> Path:
    image_path = Path(str(value))
    if image_path.is_absolute():
        return image_path.resolve()
    return (project_root / image_path).resolve()


def extract_split_features(
    split_csv_path: str | Path,
    output_csv_path: str | Path,
    project_root: str | Path | None = None,
    config: HandcraftedFeatureConfig | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Extract and save features for one existing dataset split."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    split_path = Path(split_csv_path).expanduser().resolve()
    output_path = Path(output_csv_path).expanduser().resolve()
    config = config or HandcraftedFeatureConfig()

    if not split_path.is_file():
        raise FileNotFoundError(f"Split CSV was not found: {split_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Feature file already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")

    dataframe = pd.read_csv(split_path)
    required_columns = {
        "image_id",
        "image_path",
        "class_index",
        "class_label",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{split_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if max_samples is not None:
        dataframe = dataframe.head(max_samples).copy()

    rows: list[dict[str, object]] = []
    total = len(dataframe)
    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        image_path = _resolve_image_path(root, row.image_path)
        try:
            with Image.open(image_path) as image:
                grayscale = np.array(
                    image.convert("L"),
                    dtype=np.uint8,
                    copy=True,
                )
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(
                f"Could not read image while extracting features: {image_path}"
            ) from error

        feature_values = extract_handcrafted_features(grayscale, config)
        rows.append(
            {
                "image_id": str(row.image_id),
                "image_path": str(row.image_path),
                "class_index": int(row.class_index),
                "class_label": str(row.class_label),
                **feature_values,
            }
        )

        if position % 1000 == 0 or position == total:
            print(f"Extracted {position:,}/{total:,} images")

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    result.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    print(f"Saved features: {output_path}")
    save_feature_metadata(
        feature_dataframe=result,
        split_csv_path=split_path,
        feature_csv_path=output_path,
        config=config,
        max_samples=max_samples,
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract handcrafted features from fixed data splits."
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug only: process only the first N rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = Path(__file__).resolve().parents[1]
    selected_splits = (
        tuple(SPLIT_FILENAMES)
        if arguments.split == "all"
        else (arguments.split,)
    )

    for split_name in selected_splits:
        debug_suffix = "_debug" if arguments.max_samples is not None else ""
        split_csv_path = (
            root / "data" / "splits" / SPLIT_FILENAMES[split_name]
        )
        output_csv_path = (
            root
            / "data"
            / "features"
            / f"{split_name}_features{debug_suffix}.csv"
        )
        extract_split_features(
            split_csv_path=split_csv_path,
            output_csv_path=output_csv_path,
            project_root=root,
            max_samples=arguments.max_samples,
            overwrite=arguments.overwrite,
        )


if __name__ == "__main__":
    main()
