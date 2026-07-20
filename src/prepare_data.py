"""Prepare the InfraredSolarModules dataset for 12-class model training."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# Fixed random seed to make the dataset splits reproducible
RANDOM_SEED = 42

# Official dataset repository. A shallow clone uses the normal github.com
# repository address and does not require access to raw.githubusercontent.com.
DATASET_REPOSITORY_URL = (
    "https://github.com/raptormaps/infraredsolarmodules"
)


# Keep all 12 original classes without merging
CLASS_NAMES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    Files with exactly the same content produce the same SHA-256 hash.
    The hash is used to identify exact duplicate images.

    Args:
        file_path: Path to the file.

    Returns:
        The SHA-256 hash as a hexadecimal string.
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def extract_dataset(
    raw_dir: Path,
    repository_url: str = DATASET_REPOSITORY_URL,
) -> Path:
    """
    Locate or extract the InfraredSolarModules dataset.

    The official repository is shallow-cloned into the operating system's
    temporary directory. Its dataset ZIP is extracted and the temporary clone
    is then deleted automatically. Only the extracted images and metadata
    remain in the project.

    Args:
        raw_dir: Directory where the extracted dataset is stored.
        repository_url: URL of the official GitHub repository.

    Returns:
        Path to module_metadata.json.

    Raises:
        FileNotFoundError:
            If the repository does not contain the dataset or metadata file.
        RuntimeError:
            If Git is unavailable, cloning fails, or the dataset is invalid.
    """

    # Reuse an already extracted dataset only when its metadata is readable
    # and every image referenced by it exists. This prevents a partially
    # extracted directory from being mistaken for a complete dataset.
    existing_metadata = list(
        raw_dir.rglob("module_metadata.json")
    )

    for metadata_path in existing_metadata:
        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                existing_dataset = json.load(file)

            if not isinstance(existing_dataset, dict):
                continue

            dataset_root = metadata_path.parent
            dataset_is_complete = bool(existing_dataset) and all(
                isinstance(item, dict)
                and bool(item.get("image_filepath"))
                and (
                    dataset_root / item["image_filepath"]
                ).is_file()
                for item in existing_dataset.values()
            )

            if dataset_is_complete:
                print(
                    "A complete extracted dataset was found. "
                    "Skipping download and extraction."
                )
                return metadata_path
        except (OSError, json.JSONDecodeError):
            continue

    if existing_metadata:
        print(
            "An incomplete extracted dataset was found. "
            "Downloading a fresh copy."
        )

    extracted_dir = (
        raw_dir / "extracted"
    )

    # This branch is reached only when no complete dataset was found. Remove
    # the known extraction directory before downloading so files from a failed
    # or older extraction cannot be mixed with the fresh dataset.
    if extracted_dir.exists():
        resolved_extracted_dir = extracted_dir.resolve()
        resolved_raw_dir = raw_dir.resolve()

        if (
            resolved_extracted_dir.parent != resolved_raw_dir
            or resolved_extracted_dir.name != "extracted"
        ):
            raise RuntimeError(
                "Refusing to remove an unexpected extraction directory: "
                f"{resolved_extracted_dir}"
            )

        print("Removing incomplete extracted dataset...")
        shutil.rmtree(resolved_extracted_dir)

    extracted_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Cloning dataset repository: {repository_url}")

    # Keep the repository and its ZIP outside the project. TemporaryDirectory
    # removes both automatically after extraction, including on failure.
    try:
        with tempfile.TemporaryDirectory(
            prefix="infrared_solar_modules_"
        ) as temporary_dir:
            repository_dir = (
                Path(temporary_dir) / "repository"
            )

            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    repository_url,
                    str(repository_dir),
                ],
                check=True,
            )

            archive_path = (
                repository_dir
                / "2020-02-14_InfraredSolarModules.zip"
            )

            if not archive_path.is_file():
                raise FileNotFoundError(
                    "The dataset ZIP was not found in the cloned repository."
                )

            print("Repository cloned. Extracting dataset...")

            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extracted_dir)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Git is unavailable or the cloned repository does not contain "
            "the expected dataset ZIP."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Dataset repository clone failed: {repository_url}"
        ) from error
    except zipfile.BadZipFile as error:
        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive."
        ) from error

    metadata_candidates = list(
        extracted_dir.rglob(
            "module_metadata.json"
        )
    )

    if not metadata_candidates:
        raise FileNotFoundError(
            "The dataset was extracted, but "
            "module_metadata.json could not be found."
        )

    return metadata_candidates[0]


def inspect_image(
    image_path: Path,
) -> dict[str, object]:
    """
    Inspect a single image file.

    The function checks:

    1. Whether the image file exists.
    2. Whether Pillow can open the image.
    3. Whether the image file is corrupted.
    4. The image width and height.
    5. The image channel mode.
    6. The SHA-256 file hash.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing the inspection results.
    """
    result: dict[str, object] = {
        "exists": image_path.exists(),
        "is_valid": False,
        "width": None,
        "height": None,
        "mode": None,
        "sha256": None,
        "error": "",
    }

    if not image_path.exists():
        result["error"] = "missing_file"
        return result

    try:
        with Image.open(
            image_path
        ) as image:
            width, height = image.size
            mode = image.mode

            # Verify that the image file is not corrupted
            image.verify()

        # verify() checks the container structure but does not decode all
        # pixels. Reopen the file and force a full pixel decode as a separate
        # corruption check.
        with Image.open(
            image_path
        ) as image:
            image.load()

        result.update(
            {
                "is_valid": True,
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": calculate_sha256(
                    image_path
                ),
            }
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def create_stratified_splits(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create stratified train, validation, and test splits.

    The proportions of all 12 classes are kept approximately
    equal across the three dataset splits.

    Split ratios:

    - Training: 70%
    - Validation: 15%
    - Test: 15%

    Args:
        dataframe: Clean dataset metadata.

    Returns:
        Training, validation, and test DataFrames.
    """

    # First split:
    # 70% training data and 30% temporary data
    train_dataframe, temporary_dataframe = (
        train_test_split(
            dataframe,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=dataframe[
                "class_label"
            ],
        )
    )

    # Second split:
    # divide the temporary 30% equally into
    # 15% validation and 15% test data
    (
        validation_dataframe,
        test_dataframe,
    ) = train_test_split(
        temporary_dataframe,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_dataframe[
            "class_label"
        ],
    )

    return (
        train_dataframe.reset_index(
            drop=True
        ),
        validation_dataframe.reset_index(
            drop=True
        ),
        test_dataframe.reset_index(
            drop=True
        ),
    )


def remove_previous_outputs(
    processed_dir: Path,
    splits_dir: Path,
    metrics_dir: Path,
) -> None:
    """Remove only the files generated by this preparation script."""
    generated_files = (
        processed_dir / "metadata_all.csv",
        processed_dir / "metadata_clean.csv",
        processed_dir / "label_to_index.json",
        splits_dir / "train.csv",
        splits_dir / "val.csv",
        splits_dir / "test.csv",
        metrics_dir / "data_quality_report.json",
        metrics_dir / "duplicate_label_conflicts.csv",
    )

    removed_files = 0

    for file_path in generated_files:
        if file_path.is_file():
            file_path.unlink()
            removed_files += 1

    print(
        f"Removed {removed_files} previous generated "
        "file(s)."
    )


def validate_split_integrity(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Ensure that no image content appears in more than one split."""
    split_hashes = {
        "train": set(train_dataframe["sha256"]),
        "validation": set(validation_dataframe["sha256"]),
        "test": set(test_dataframe["sha256"]),
    }
    overlapping_pairs = []

    for first_name, second_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap_count = len(
            split_hashes[first_name]
            & split_hashes[second_name]
        )

        if overlap_count:
            overlapping_pairs.append(
                f"{first_name}/{second_name}: {overlap_count}"
            )

    if overlapping_pairs:
        raise ValueError(
            "Dataset split overlap detected by SHA-256 hash: "
            + ", ".join(overlapping_pairs)
        )

    print("Dataset split overlap check passed.")


def main() -> None:
    """Run the complete dataset preparation pipeline."""

    # prepare_data.py is located inside project/src/.
    # parents[1] therefore points to the project root directory.
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    def display_path(path: Path) -> str:
        """Return a portable project-relative path for console output."""
        return path.relative_to(project_root).as_posix()

    raw_dir = (
        project_root / "data" / "raw"
    )

    processed_dir = (
        project_root / "data" / "processed"
    )

    splits_dir = (
        project_root / "data" / "splits"
    )

    metrics_dir = (
        project_root / "results" / "metrics"
    )

    # Create all required output directories
    for directory in (
        raw_dir,
        processed_dir,
        splits_dir,
        metrics_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "Project root: ."
    )

    print(
        f"Raw data directory: {display_path(raw_dir)}"
    )

    # Extract the dataset or locate an existing extracted copy
    metadata_path = extract_dataset(
        raw_dir
    )

    # The directory containing module_metadata.json
    # is treated as the dataset root
    dataset_root = metadata_path.parent

    print(
        f"\nMetadata file: {display_path(metadata_path)}"
    )

    print(
        f"Dataset root: {display_path(dataset_root)}\n"
    )

    # The source dataset is known to be available at this point. Remove only
    # this script's previous derived outputs so stale CSV/JSON files cannot be
    # mistaken for results from the current run. Raw images are never removed.
    remove_previous_outputs(
        processed_dir,
        splits_dir,
        metrics_dir,
    )

    # Load the original JSON metadata
    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "The top-level structure of "
            "module_metadata.json must be a dictionary."
        )

    rows: list[
        dict[str, object]
    ] = []

    total_images = len(metadata)

    # Inspect every image in the dataset
    for index, (
        image_id,
        item,
    ) in enumerate(
        metadata.items(),
        start=1,
    ):
        original_label = item[
            "anomaly_class"
        ]

        # Confirm that the label belongs to the 12 known classes
        if original_label not in CLASS_NAMES:
            raise ValueError(
                "An undefined class was found: "
                f"{original_label}"
            )

        relative_path = Path(
            item["image_filepath"]
        )

        image_path = (
            dataset_root / relative_path
        )

        inspection_result = inspect_image(
            image_path
        )

        rows.append(
            {
                "image_id": str(
                    image_id
                ),
                # Store a project-relative path so generated CSV files remain
                # portable when the project is moved to another computer.
                # Consumers should resolve it as: project_root / image_path.
                "image_path": (
                    image_path.relative_to(
                        project_root
                    ).as_posix()
                ),
                "relative_path": (
                    relative_path.as_posix()
                ),
                "original_label": (
                    original_label
                ),

                # No class merging is performed.
                # The training class is the original class.
                "class_label": (
                    original_label
                ),

                **inspection_result,
            }
        )

        # Display progress after every 2,000 images
        if (
            index % 2000 == 0
            or index == total_images
        ):
            print(
                f"Inspected "
                f"{index:,}/{total_images:,} images"
            )

    # Convert all image records into a Pandas DataFrame
    all_dataframe = pd.DataFrame(
        rows
    )

    # Use only valid images when analysing duplicate files
    valid_dataframe = all_dataframe[
        all_dataframe["is_valid"]
    ].copy()

    # Count how many times each SHA-256 hash appears
    duplicate_counts = (
        valid_dataframe[
            "sha256"
        ].value_counts()
    )

    # Hashes appearing more than once represent exact duplicates
    duplicate_hashes = set(
        duplicate_counts[
            duplicate_counts > 1
        ].index
    )

    # Mark every record belonging to an exact duplicate group
    all_dataframe[
        "is_exact_duplicate"
    ] = all_dataframe[
        "sha256"
    ].isin(
        duplicate_hashes
    )

    # Count the number of different labels assigned to each hash
    labels_per_hash = (
        valid_dataframe
        .groupby(
            "sha256"
        )[
            "original_label"
        ]
        .nunique()
    )

    # A hash with more than one label indicates a label conflict
    conflict_hashes = set(
        labels_per_hash[
            labels_per_hash > 1
        ].index
    )

    # Mark all records involved in label conflicts
    all_dataframe[
        "has_label_conflict"
    ] = all_dataframe[
        "sha256"
    ].isin(
        conflict_hashes
    )

    # Clean the dataset:
    #
    # 1. Keep only valid images.
    # 2. Remove all images involved in label conflicts.
    # 3. Keep only one record for each exact duplicate image.
    clean_dataframe = all_dataframe[
        all_dataframe["is_valid"]
        & ~all_dataframe[
            "has_label_conflict"
        ]
    ].copy()

    clean_dataframe = (
        clean_dataframe
        .drop_duplicates(
            subset="sha256",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # Create a fixed mapping from class names to numeric indices
    label_to_index = {
        class_name: index
        for index, class_name
        in enumerate(CLASS_NAMES)
    }

    # Convert the 12 text class labels into integer class indices
    clean_dataframe[
        "class_index"
    ] = clean_dataframe[
        "class_label"
    ].map(
        label_to_index
    )

    # Confirm that every class received a numeric index
    if clean_dataframe[
        "class_index"
    ].isna().any():
        missing_labels = (
            clean_dataframe.loc[
                clean_dataframe[
                    "class_index"
                ].isna(),
                "class_label",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Some classes do not have a "
            "numeric index: "
            f"{missing_labels}"
        )

    # PyTorch classification targets must use integer labels
    clean_dataframe[
        "class_index"
    ] = clean_dataframe[
        "class_index"
    ].astype(int)

    # Create stratified train, validation, and test datasets
    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_stratified_splits(
        clean_dataframe
    )

    # Guard against train/validation/test leakage using image content hashes,
    # rather than relying only on filenames or row indices.
    validate_split_integrity(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    # Save metadata for every original image record
    all_dataframe.to_csv(
        processed_dir
        / "metadata_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save cleaned metadata for EDA and model training
    clean_dataframe.to_csv(
        processed_dir
        / "metadata_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save all records involved in duplicate-image label conflicts
    (
        all_dataframe[
            all_dataframe[
                "has_label_conflict"
            ]
        ]
        .sort_values(
            "sha256"
        )
        .to_csv(
            metrics_dir
            / "duplicate_label_conflicts.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    # Save the fixed training split
    train_dataframe.to_csv(
        splits_dir
        / "train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save the fixed validation split
    validation_dataframe.to_csv(
        splits_dir
        / "val.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save the fixed test split
    test_dataframe.to_csv(
        splits_dir
        / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save the class-name-to-index mapping used by the model
    with (
        processed_dir
        / "label_to_index.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_to_index,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Summarise image dimensions
    image_size_counts = {
        f"{int(width)}x{int(height)}": int(
            count
        )
        for (
            width,
            height,
        ), count in (
            valid_dataframe
            .groupby(
                ["width", "height"]
            )
            .size()
            .items()
        )
    }

    # Summarise image channel modes
    image_mode_counts = {
        str(mode): int(count)
        for mode, count in (
            valid_dataframe[
                "mode"
            ]
            .value_counts()
            .items()
        )
    }

    # Summarise the original 12-class distribution
    original_class_counts = {
        str(label): int(count)
        for label, count in (
            all_dataframe[
                "original_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    # Summarise the cleaned 12-class distribution
    clean_class_counts = {
        str(label): int(count)
        for label, count in (
            clean_dataframe[
                "class_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    # Summarise the class distribution in each dataset split
    train_class_counts = {
        str(label): int(count)
        for label, count in (
            train_dataframe[
                "class_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    validation_class_counts = {
        str(label): int(count)
        for label, count in (
            validation_dataframe[
                "class_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    test_class_counts = {
        str(label): int(count)
        for label, count in (
            test_dataframe[
                "class_label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    # Create the complete data quality report
    report = {
        "classification_task": (
            "12-class classification"
        ),
        "number_of_classes": len(
            CLASS_NAMES
        ),
        "class_names": CLASS_NAMES,
        "total_records": int(
            len(all_dataframe)
        ),
        "valid_images": int(
            all_dataframe[
                "is_valid"
            ].sum()
        ),
        "invalid_or_missing_images": int(
            (
                ~all_dataframe[
                    "is_valid"
                ]
            ).sum()
        ),
        "exact_duplicate_groups": int(
            len(duplicate_hashes)
        ),
        "label_conflict_groups": int(
            len(conflict_hashes)
        ),
        "clean_unique_images": int(
            len(clean_dataframe)
        ),
        "image_sizes": (
            image_size_counts
        ),
        "image_modes": (
            image_mode_counts
        ),
        "original_class_counts": (
            original_class_counts
        ),
        "clean_class_counts": (
            clean_class_counts
        ),
        "split_sizes": {
            "train": int(
                len(train_dataframe)
            ),
            "validation": int(
                len(validation_dataframe)
            ),
            "test": int(
                len(test_dataframe)
            ),
        },
        "split_class_counts": {
            "train": (
                train_class_counts
            ),
            "validation": (
                validation_class_counts
            ),
            "test": (
                test_class_counts
            ),
        },
    }

    # Save the data quality report as JSON
    with (
        metrics_dir
        / "data_quality_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n========== "
        "Data preparation completed "
        "=========="
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nGenerated files:"
    )

    print(
        display_path(
            processed_dir / "metadata_all.csv"
        )
    )

    print(
        display_path(
            processed_dir / "metadata_clean.csv"
        )
    )

    print(
        display_path(
            processed_dir / "label_to_index.json"
        )
    )

    print(
        display_path(
            splits_dir / "train.csv"
        )
    )

    print(
        display_path(
            splits_dir / "val.csv"
        )
    )

    print(
        display_path(
            splits_dir / "test.csv"
        )
    )

    print(
        display_path(
            metrics_dir / "data_quality_report.json"
        )
    )

    print(
        display_path(
            metrics_dir / "duplicate_label_conflicts.csv"
        )
    )


if __name__ == "__main__":
    main()
