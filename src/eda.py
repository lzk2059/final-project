"""Run exploratory data analysis for the InfraredSolarModules dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_SEED = 42
FIGURE_DPI = 180


def load_project_data(
    project_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    list[str],
]:
    """Load prepared metadata, dataset splits, and the fixed class order."""
    processed_dir = project_root / "data" / "processed"
    splits_dir = project_root / "data" / "splits"

    required_files = {
        "all metadata": processed_dir / "metadata_all.csv",
        "clean metadata": processed_dir / "metadata_clean.csv",
        "label mapping": processed_dir / "label_to_index.json",
        "training split": splits_dir / "train.csv",
        "validation split": splits_dir / "val.csv",
        "test split": splits_dir / "test.csv",
    }
    missing_files = [
        str(path.relative_to(project_root))
        for path in required_files.values()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Prepared dataset files are missing. Run "
            "'python src/prepare_data.py' first. Missing: "
            + ", ".join(missing_files)
        )

    all_dataframe = pd.read_csv(required_files["all metadata"])
    clean_dataframe = pd.read_csv(required_files["clean metadata"])
    splits = {
        "Train": pd.read_csv(required_files["training split"]),
        "Validation": pd.read_csv(required_files["validation split"]),
        "Test": pd.read_csv(required_files["test split"]),
    }

    with required_files["label mapping"].open(
        "r", encoding="utf-8"
    ) as file:
        label_to_index = json.load(file)

    class_names = [
        class_name
        for class_name, _ in sorted(
            label_to_index.items(),
            key=lambda item: item[1],
        )
    ]

    return all_dataframe, clean_dataframe, splits, class_names


def save_class_distribution(
    all_dataframe: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot original and cleaned class counts."""
    original_counts = (
        all_dataframe["original_label"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )
    clean_counts = (
        clean_dataframe["class_label"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )
    positions = np.arange(len(class_names))
    width = 0.4

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.bar(
        positions - width / 2,
        original_counts,
        width,
        label="Original",
        color="#4472C4",
    )
    axis.bar(
        positions + width / 2,
        clean_counts,
        width,
        label="Clean",
        color="#ED7D31",
    )
    axis.set_title("Class Distribution Before and After Cleaning")
    axis.set_xlabel("Anomaly class")
    axis.set_ylabel("Number of images")
    axis.set_xticks(positions, class_names, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_split_distribution(
    splits: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Compare class percentages across train, validation, and test splits."""
    positions = np.arange(len(class_names))
    width = 0.25
    colors = ("#4472C4", "#70AD47", "#ED7D31")

    figure, axis = plt.subplots(figsize=(14, 7))

    for offset, ((split_name, dataframe), color) in enumerate(
        zip(splits.items(), colors, strict=True)
    ):
        percentages = (
            dataframe["class_label"]
            .value_counts(normalize=True)
            .reindex(class_names, fill_value=0)
            * 100
        )
        axis.bar(
            positions + (offset - 1) * width,
            percentages,
            width,
            label=split_name,
            color=color,
        )

    axis.set_title("Class Proportions Across Dataset Splits")
    axis.set_xlabel("Anomaly class")
    axis.set_ylabel("Percentage of split (%)")
    axis.set_xticks(positions, class_names, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def analyse_images(
    clean_dataframe: pd.DataFrame,
    class_names: list[str],
    project_root: Path,
) -> tuple[
    dict[str, list[float]],
    np.ndarray,
]:
    """Calculate per-image mean intensity and the pixel histogram."""
    image_intensities = {class_name: [] for class_name in class_names}
    pixel_histogram = np.zeros(256, dtype=np.int64)

    total_images = len(clean_dataframe)

    for index, row in enumerate(
        clean_dataframe.itertuples(index=False),
        start=1,
    ):
        image_path = project_root / Path(row.image_path)

        with Image.open(image_path) as image:
            grayscale_image = image.convert("L")
            image_array = np.asarray(grayscale_image, dtype=np.uint8)

        class_name = str(row.class_label)
        float_image = image_array.astype(np.float64)

        image_intensities[class_name].append(float(float_image.mean()))
        pixel_histogram += np.bincount(
            image_array.ravel(),
            minlength=256,
        )

        if index % 2000 == 0 or index == total_images:
            print(f"Analysed {index:,}/{total_images:,} images")

    return image_intensities, pixel_histogram


def save_class_samples(
    clean_dataframe: pd.DataFrame,
    class_names: list[str],
    project_root: Path,
    output_path: Path,
) -> None:
    """Save one reproducibly selected example from every class."""
    figure, axes = plt.subplots(3, 4, figsize=(12, 10))

    for axis, class_name in zip(
        axes.flat,
        class_names,
        strict=True,
    ):
        class_rows = clean_dataframe[
            clean_dataframe["class_label"] == class_name
        ]
        sample = class_rows.sample(
            n=1,
            random_state=RANDOM_SEED,
        ).iloc[0]
        image_path = project_root / Path(sample["image_path"])

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("L"))

        axis.imshow(image_array, cmap="inferno")
        axis.set_title(class_name)
        axis.axis("off")

    figure.suptitle("Representative Infrared Image from Each Class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_pixel_intensity_distribution(
    pixel_histogram: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the training split's overall pixel-intensity distribution."""
    figure, axis = plt.subplots(figsize=(9, 6))
    intensity_values = np.arange(256)
    pixel_percentages = pixel_histogram / pixel_histogram.sum() * 100

    axis.plot(
        intensity_values,
        pixel_percentages,
        color="#4472C4",
        linewidth=1.5,
    )
    axis.fill_between(
        intensity_values,
        pixel_percentages,
        color="#4472C4",
        alpha=0.2,
    )
    axis.set_title("Training Pixel-Intensity Distribution")
    axis.set_xlabel("Grayscale intensity")
    axis.set_ylabel("Percentage of pixels (%)")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_class_intensity_boxplot(
    image_intensities: dict[str, list[float]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot training-image mean-intensity variation for every class."""
    figure, axis = plt.subplots(figsize=(12, 7))

    axis.boxplot(
        [image_intensities[name] for name in class_names],
        tick_labels=class_names,
        showfliers=False,
    )
    axis.set_title("Training Mean Image Intensity by Class")
    axis.set_xlabel("Anomaly class")
    axis.set_ylabel("Mean grayscale intensity")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def create_eda_summary(
    all_dataframe: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    class_names: list[str],
    image_intensities: dict[str, list[float]],
) -> dict[str, object]:
    """Create a machine-readable summary of the main EDA findings."""
    original_counts = (
        all_dataframe["original_label"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )
    clean_counts = (
        clean_dataframe["class_label"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )
    split_counts = {
        split_name.lower(): {
            class_name: int(count)
            for class_name, count in (
                dataframe["class_label"]
                .value_counts()
                .reindex(class_names, fill_value=0)
                .items()
            )
        }
        for split_name, dataframe in splits.items()
    }
    split_hashes = {
        name: set(dataframe["sha256"])
        for name, dataframe in splits.items()
    }
    split_overlap = {
        "train_validation": len(
            split_hashes["Train"] & split_hashes["Validation"]
        ),
        "train_test": len(
            split_hashes["Train"] & split_hashes["Test"]
        ),
        "validation_test": len(
            split_hashes["Validation"] & split_hashes["Test"]
        ),
    }
    class_intensity_summary = {
        class_name: {
            "mean": float(np.mean(image_intensities[class_name])),
            "standard_deviation": float(
                np.std(image_intensities[class_name])
            ),
            "median": float(np.median(image_intensities[class_name])),
        }
        for class_name in class_names
    }

    return {
        "random_seed": RANDOM_SEED,
        "total_original_records": int(len(all_dataframe)),
        "total_clean_records": int(len(clean_dataframe)),
        "removed_records": int(len(all_dataframe) - len(clean_dataframe)),
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "original_class_counts": {
            name: int(count) for name, count in original_counts.items()
        },
        "clean_class_counts": {
            name: int(count) for name, count in clean_counts.items()
        },
        "class_imbalance_ratio": float(
            clean_counts.max() / clean_counts.min()
        ),
        "majority_class": str(clean_counts.idxmax()),
        "minority_class": str(clean_counts.idxmin()),
        "image_sizes": sorted(
            {
                f"{int(width)}x{int(height)}"
                for width, height in zip(
                    clean_dataframe["width"],
                    clean_dataframe["height"],
                    strict=True,
                )
            }
        ),
        "image_modes": {
            str(mode): int(count)
            for mode, count in clean_dataframe["mode"].value_counts().items()
        },
        "split_sizes": {
            name.lower(): int(len(dataframe))
            for name, dataframe in splits.items()
        },
        "split_class_counts": split_counts,
        "split_sha256_overlap": split_overlap,
        "intensity_analysis_source": "training_split",
        "training_class_mean_intensity": class_intensity_summary,
    }


def main() -> None:
    """Run the complete exploratory data analysis pipeline."""
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / "results" / "figures"
    metrics_dir = project_root / "results" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_dataframe, clean_dataframe, splits, class_names = (
        load_project_data(project_root)
    )

    print(f"Loaded {len(all_dataframe):,} original records.")
    print(f"Loaded {len(clean_dataframe):,} clean records.")

    save_class_distribution(
        all_dataframe,
        clean_dataframe,
        class_names,
        figures_dir / "eda_class_distribution.png",
    )
    save_split_distribution(
        splits,
        class_names,
        figures_dir / "eda_split_distribution.png",
    )
    save_class_samples(
        splits["Train"],
        class_names,
        project_root,
        figures_dir / "eda_class_samples.png",
    )

    image_intensities, pixel_histogram = analyse_images(
        splits["Train"],
        class_names,
        project_root,
    )
    save_pixel_intensity_distribution(
        pixel_histogram,
        figures_dir / "eda_pixel_intensity_distribution.png",
    )
    save_class_intensity_boxplot(
        image_intensities,
        class_names,
        figures_dir / "eda_class_intensity_boxplot.png",
    )

    summary = create_eda_summary(
        all_dataframe,
        clean_dataframe,
        splits,
        class_names,
        image_intensities,
    )
    summary_path = metrics_dir / "eda_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nEDA completed. Generated files:")

    for output_path in (
        figures_dir / "eda_class_distribution.png",
        figures_dir / "eda_split_distribution.png",
        figures_dir / "eda_class_samples.png",
        figures_dir / "eda_pixel_intensity_distribution.png",
        figures_dir / "eda_class_intensity_boxplot.png",
        summary_path,
    ):
        print(output_path.relative_to(project_root).as_posix())


if __name__ == "__main__":
    main()
