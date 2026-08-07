"""Generate Grad-CAM overlays from saved inference predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import ROOT, load_classifier, load_image_tensor, portable_path
from src.preprocessing import (
    PreprocessingConfig,
    calculate_resize_with_padding_geometry,
)


DEFAULT_INPUT_DIR = ROOT / "results" / "inference"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "gradcam"


def resolve_saved_path(value: object) -> Path:
    """Resolve project-relative paths stored by inference.py."""

    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def target_layer(model: nn.Module, model_name: str) -> tuple[nn.Module, str]:
    """Return the final spatial feature layer for a supported classifier."""

    normalized = model_name.lower().replace("-", "_")
    if normalized in {"efficientnet_b0", "efficientnet_b0_multiscale"}:
        return model.features[-1], "features[-1]"
    if normalized in {"resnet18", "resnet_18"}:
        return model.layer4[-1], "layer4[-1]"
    raise ValueError(f"Grad-CAM does not support model: {model_name}")


def calculate_gradcam(
    model: nn.Module,
    layer: nn.Module,
    tensor: torch.Tensor,
    target_index: int,
) -> tuple[np.ndarray, int]:
    """Calculate a normalized Grad-CAM map and the current predicted index."""

    captured: dict[str, torch.Tensor] = {}

    def capture_activation(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        predicted_index = int(logits.argmax(dim=1).item())
        if not 0 <= target_index < logits.shape[1]:
            raise ValueError(f"Target class index is out of range: {target_index}")
        logits[0, target_index].backward()
    finally:
        handle.remove()

    if "activation" not in captured or "gradient" not in captured:
        raise RuntimeError("Target layer did not produce Grad-CAM tensors.")
    activation = captured["activation"].detach()
    gradient = captured["gradient"].detach()
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise RuntimeError("Invalid Grad-CAM activation or gradient shape.")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    minimum, maximum = cam.min(), cam.max()
    if not torch.isfinite(cam).all() or float(maximum - minimum) <= 1e-12:
        raise RuntimeError("Grad-CAM map is empty or non-finite.")
    return ((cam - minimum) / (maximum - minimum)).cpu().numpy(), predicted_index


def save_overlay(
    image_path: Path,
    cam: np.ndarray,
    output_path: Path,
    predicted_class: str,
    confidence: float,
) -> None:
    """Overlay model attention on the original grayscale infrared image."""

    with Image.open(image_path) as source:
        grayscale_image = source.convert("L")
        grayscale = np.asarray(grayscale_image, dtype=np.float32) / 255.0
        resized_width, resized_height, left, top, _, _ = (
            calculate_resize_with_padding_geometry(
                grayscale_image.size,
                PreprocessingConfig().image_size,
            )
        )
        cam = cam[
            top : top + resized_height,
            left : left + resized_width,
        ]
        if cam.shape != (resized_height, resized_width):
            raise RuntimeError("Failed to remove Grad-CAM padding correctly.")
        cam = np.asarray(
            Image.fromarray(np.uint8(cam * 255)).resize(
                grayscale_image.size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    figure, axis = plt.subplots(figsize=(5, 7))
    axis.imshow(grayscale, cmap="gray", interpolation="nearest")
    axis.imshow(cam, cmap="jet", alpha=0.45, interpolation="bilinear")
    axis.set_title(f"{predicted_class} | confidence {confidence:.4f}")
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def explain_prediction(
    prediction_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Explain one successful inference JSON and save its Grad-CAM overlay."""

    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction.get("status") != "success":
        raise ValueError("Prediction status is not success.")
    image_path = resolve_saved_path(prediction["image"])
    checkpoint_path = resolve_saved_path(prediction["checkpoint"])
    model, device, names, model_name, _ = load_classifier(checkpoint_path)
    if prediction.get("class_names") != names:
        raise ValueError("Prediction class mapping does not match the model contract.")
    layer, layer_name = target_layer(model, model_name)
    cam, current_prediction = calculate_gradcam(
        model,
        layer,
        load_image_tensor(image_path, device),
        int(prediction["predicted_index"]),
    )
    if current_prediction != int(prediction["predicted_index"]):
        raise RuntimeError("Current model prediction differs from the saved inference.")
    output_path = output_dir / prediction_path.name.replace(
        "_prediction.json", "_gradcam_overlay.png"
    )
    save_overlay(
        image_path,
        cam,
        output_path,
        str(prediction["predicted_class"]),
        float(prediction["confidence"]),
    )
    return {
        "status": "success",
        "prediction": prediction_path.name,
        "image": str(prediction["image"]),
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "target_layer": layer_name,
        "overlay": portable_path(output_path),
        "error": "",
    }


def run_explanations(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, object]]:
    """Explain every saved prediction while isolating per-case failures."""

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    predictions = sorted(input_path.glob("*_prediction.json"))
    if not predictions:
        raise FileNotFoundError(f"No inference prediction JSON files found in: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    records = []
    for index, prediction_path in enumerate(predictions, start=1):
        try:
            record = explain_prediction(prediction_path, output_path)
            message = str(record["overlay"])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            record = {
                "status": "error",
                "prediction": prediction_path.name,
                "image": "",
                "predicted_class": "",
                "confidence": "",
                "target_layer": "",
                "overlay": "",
                "error": f"{type(error).__name__}: {error}",
            }
            message = str(record["error"])
        records.append(record)
        print(f"[{index}/{len(predictions)}] {prediction_path.name}: {message}")

    fields = [
        "status",
        "prediction",
        "image",
        "predicted_class",
        "confidence",
        "target_layer",
        "overlay",
        "error",
    ]
    with (output_path / "explanations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Grad-CAM results: {output_path}")
    return records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain saved inference predictions.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_explanations(arguments.input_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
