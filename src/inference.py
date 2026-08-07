"""Predict infrared images placed in the inference input directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.model import create_model
from src.preprocessing import build_evaluation_preprocessing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "efficientnet_b0_multiscale_cross_entropy_seed42_best.pt"
)
DEFAULT_INPUT_DIR = ROOT / "data" / "inference_images"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "inference"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = (
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
)


def load_class_names(num_classes: int) -> list[str]:
    """Load class names in numeric-index order."""

    path = ROOT / "data" / "processed" / "label_to_index.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise TypeError("label_to_index.json must contain a mapping.")
    if num_classes != len(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint has {num_classes} classes; expected {len(CLASS_NAMES)}."
        )
    names = [""] * num_classes
    for name, index in mapping.items():
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_classes
            or names[index]
        ):
            raise ValueError(f"Invalid class mapping for '{name}': {index}")
        names[index] = str(name)
    if any(not name for name in names):
        raise ValueError("Class mapping is incomplete.")
    if tuple(names) != CLASS_NAMES:
        raise ValueError("Label mapping does not match the trained model class contract.")
    return list(CLASS_NAMES)


def load_classifier(
    checkpoint: str | Path,
) -> tuple[nn.Module, torch.device, list[str], str, Path]:
    """Load a trained classifier for inference or Grad-CAM."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"PyTorch checkpoint was not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location=device, weights_only=True)
    model_name = str(saved["model_name"])
    num_classes = int(saved["num_classes"])
    model = create_model(model_name, num_classes, pretrained=False).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, device, load_class_names(num_classes), model_name, path


def load_image_tensor(image: str | Path, device: torch.device) -> torch.Tensor:
    """Read one image using the deterministic evaluation preprocessing."""

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image was not found: {path}")
    with Image.open(path) as source:
        return build_evaluation_preprocessing()(source.copy()).unsqueeze(0).to(device)


def predict_probabilities(
    model: nn.Module, image: str | Path, device: torch.device
) -> torch.Tensor:
    """Return a valid one-dimensional probability vector."""

    with torch.inference_mode():
        probabilities = torch.softmax(model(load_image_tensor(image, device)), dim=1)[0]
    probabilities = probabilities.cpu()
    if not torch.isfinite(probabilities).all() or not torch.isclose(
        probabilities.sum(), torch.tensor(1.0), atol=1e-5
    ):
        raise RuntimeError("Model produced invalid probabilities.")
    return probabilities


def prediction_record(
    image: Path,
    checkpoint: Path,
    model_name: str,
    device: torch.device,
    names: list[str],
    probabilities: torch.Tensor,
) -> dict[str, object]:
    """Build the saved record later consumed by explain.py."""

    predicted = int(probabilities.argmax().item())
    return {
        "status": "success",
        "image": str(image.resolve()),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "model_name": model_name,
        "class_names": names,
        "predicted_index": predicted,
        "predicted_class": names[predicted],
        "confidence": float(probabilities[predicted]),
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(names)
        },
    }


def portable_path(path: Path) -> str:
    """Use project-relative paths when possible so results survive relocation."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def result_filename(image: Path) -> str:
    """Keep the image extension in the result name to prevent stem collisions."""

    return f"{image.name.replace('.', '_')}_prediction.json"


def run_inference(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> list[dict[str, object]]:
    """Predict every supported image and save JSON plus one summary CSV."""

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    input_path.mkdir(parents=True, exist_ok=True)
    images = sorted(
        (path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise FileNotFoundError(f"No images found. Place images in: {input_path}")

    model, device, names, model_name, checkpoint_path = load_classifier(checkpoint)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, image in enumerate(images, start=1):
        try:
            record = prediction_record(
                image,
                checkpoint_path,
                model_name,
                device,
                names,
                predict_probabilities(model, image, device),
            )
            record["image"] = portable_path(image)
            record["checkpoint"] = portable_path(checkpoint_path)
            message = (
                f"{record['predicted_class']} ({record['confidence']:.4f})"
            )
        except (OSError, RuntimeError, ValueError) as error:
            record = {
                "status": "error",
                "image": portable_path(image),
                "checkpoint": portable_path(checkpoint_path),
                "error": f"{type(error).__name__}: {error}",
            }
            message = str(record["error"])
        records.append(record)
        result_path = output_path / result_filename(image)
        result_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{index}/{len(images)}] {image.name}: {message}")

    probability_columns = [f"probability_{name}" for name in names]
    with (output_path / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "status",
                "image",
                "predicted_class",
                "confidence",
                "error",
                *probability_columns,
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "status": record["status"],
                    "image": record["image"],
                    "predicted_class": record.get("predicted_class", ""),
                    "confidence": record.get("confidence", ""),
                    "error": record.get("error", ""),
                    **{
                        f"probability_{name}": probability
                        for name, probability in record.get("probabilities", {}).items()
                    },
                }
            )
    print(f"Results: {output_path}")
    return records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict all inference images.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_inference(arguments.input_dir, arguments.output_dir, arguments.checkpoint)


if __name__ == "__main__":
    main()
