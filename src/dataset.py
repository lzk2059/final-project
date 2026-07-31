"""PyTorch datasets and data loaders for infrared image classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import (
    PreprocessingConfig,
    build_evaluation_preprocessing,
    build_train_preprocessing,
)


NUMBER_OF_CLASSES: Final[int] = 12
REQUIRED_COLUMNS: Final[set[str]] = {
    "image_path",
    "class_index",
    "class_label",
}


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration shared by the training, validation, and test loaders."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    drop_last_training_batch: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero."
            )


class InfraredDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load infrared images and their integer class labels from a split CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path | None = None,
        training: bool = False,
        preprocessing_config: PreprocessingConfig | None = None,
        validate_files: bool = True,
    ) -> None:
        """
        Create a dataset for one fixed data split.

        Args:
            csv_path:
                Path to ``train.csv``, ``val.csv``, or ``test.csv``.
            project_root:
                Root used to resolve the project-relative ``image_path``
                stored in the CSV. It is detected automatically when omitted.
            training:
                Whether to use random training augmentation.
            preprocessing_config:
                Optional shared image preprocessing configuration.
            validate_files:
                Check every referenced image path during initialization.
        """

        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset split CSV was not found: {self.csv_path}"
            )

        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.training = training

        self.dataframe = pd.read_csv(self.csv_path)
        self._validate_dataframe()

        self.image_paths = [
            self._resolve_image_path(value)
            for value in self.dataframe["image_path"].tolist()
        ]
        self.labels = (
            self.dataframe["class_index"].astype(int).tolist()
        )
        self.class_labels = (
            self.dataframe["class_label"].astype(str).tolist()
        )

        if validate_files:
            missing_paths = [
                path for path in self.image_paths if not path.is_file()
            ]
            if missing_paths:
                preview = ", ".join(
                    str(path) for path in missing_paths[:5]
                )
                raise FileNotFoundError(
                    f"{len(missing_paths)} image file(s) referenced by "
                    f"{self.csv_path.name} were not found. Examples: {preview}"
                )

        if training:
            self.transform = build_train_preprocessing(
                preprocessing_config
            )
        else:
            self.transform = build_evaluation_preprocessing(
                preprocessing_config
            )

    def _validate_dataframe(self) -> None:
        """Validate required columns and the 12-class target range."""

        missing_columns = REQUIRED_COLUMNS - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{self.csv_path.name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if self.dataframe.empty:
            raise ValueError(f"{self.csv_path.name} contains no samples.")
        if self.dataframe[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(
                f"{self.csv_path.name} contains missing required values."
            )

        numeric_labels = pd.to_numeric(
            self.dataframe["class_index"],
            errors="coerce",
        )
        if numeric_labels.isna().any():
            raise ValueError(
                f"{self.csv_path.name} contains non-numeric class indices."
            )
        if not (numeric_labels == numeric_labels.astype(int)).all():
            raise ValueError(
                f"{self.csv_path.name} contains non-integer class indices."
            )
        if not numeric_labels.between(
            0,
            NUMBER_OF_CLASSES - 1,
        ).all():
            invalid = sorted(
                numeric_labels[
                    ~numeric_labels.between(
                        0,
                        NUMBER_OF_CLASSES - 1,
                    )
                ].unique().tolist()
            )
            raise ValueError(
                f"{self.csv_path.name} contains class indices outside "
                f"0-{NUMBER_OF_CLASSES - 1}: {invalid}"
            )

    def _resolve_image_path(self, value: object) -> Path:
        """Resolve a CSV image path without depending on the working directory."""

        image_path = Path(str(value))
        if image_path.is_absolute():
            return image_path.resolve()
        return (self.project_root / image_path).resolve()

    def __len__(self) -> int:
        """Return the number of images in this split."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load, preprocess, and return one image and its class index."""

        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            with Image.open(image_path) as image:
                # Copy the decoded grayscale image before the context manager
                # closes its underlying file.
                image = image.convert("L").copy()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not load image at dataset index {index}: "
                f"{image_path}"
            ) from error

        image_tensor = self.transform(image)
        return image_tensor, label

    def get_sample_info(self, index: int) -> dict[str, object]:
        """Return metadata useful for evaluation reports and error analysis."""

        return {
            "index": index,
            "image_path": str(self.image_paths[index]),
            "class_index": self.labels[index],
            "class_label": self.class_labels[index],
        }


def create_dataloader(
    dataset: InfraredDataset,
    config: DataLoaderConfig | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a loader whose behavior matches the dataset split."""

    config = config or DataLoaderConfig()
    pin_memory = (
        torch.cuda.is_available()
        if config.pin_memory is None
        else config.pin_memory
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=dataset.training,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=(
            config.drop_last_training_batch
            if dataset.training
            else False
        ),
    )


def create_dataloaders(
    project_root: str | Path | None = None,
    loader_config: DataLoaderConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    validate_files: bool = True,
) -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Create the train, validation, and test data loaders."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    splits_directory = root / "data" / "splits"
    split_definitions = {
        "train": (splits_directory / "train.csv", True),
        "validation": (splits_directory / "val.csv", False),
        "test": (splits_directory / "test.csv", False),
    }

    loaders: dict[
        str,
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ] = {}

    for split_name, (csv_path, training) in split_definitions.items():
        dataset = InfraredDataset(
            csv_path=csv_path,
            project_root=root,
            training=training,
            preprocessing_config=preprocessing_config,
            validate_files=validate_files,
        )
        loaders[split_name] = create_dataloader(
            dataset,
            config=loader_config,
        )

    return loaders
