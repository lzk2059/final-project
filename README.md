# Infrared Photovoltaic Module Fault Classification

This project implements a reproducible 12-class classification workflow for
infrared images of photovoltaic modules. It includes automatic dataset
download and preparation, deep-learning and traditional machine-learning
baselines, EfficientNet-B0 ablation experiments, multi-seed evaluation,
single-image inference, maintenance-priority recommendations, and Grad-CAM
explanations.

## Classification task

The 12 classes are:

`No-Anomaly`, `Cell`, `Cell-Multi`, `Cracking`, `Diode`, `Diode-Multi`,
`Hot-Spot`, `Hot-Spot-Multi`, `Offline-Module`, `Shadowing`, `Soiling`, and
`Vegetation`.

The main experiments are:

- ResNet-18 + Cross-Entropy
- EfficientNet-B0 + Cross-Entropy
- EfficientNet-B0 + multi-scale feature fusion + Cross-Entropy
- EfficientNet-B0 + class-balanced Focal Loss
- EfficientNet-B0 + multi-scale feature fusion + class-balanced Focal Loss
- Random Forest with balanced class weights using intensity, LBP, and GLCM
  features

Deep-learning models use AdamW, a maximum of 60 epochs, early stopping with a
patience of 8, and selection by validation Macro-F1. Formal experiments use
random seeds 42, 123, and 2026.

## Repository layout

```text
configs/                 Experiment and maintenance-rule configurations
data/inference_images/   Infrared images supplied for inference
data/processed/          Clean metadata and class mapping
data/raw/                Automatically downloaded and extracted dataset
data/splits/             Fixed train/validation/test CSV files
results/comparison/      Multi-seed summary tables
results/evaluation/      Test/validation metrics and per-image predictions
results/figures/         Training, evaluation, ablation, and comparison plots
results/gradcam/         Grad-CAM overlays and case reports
results/inference/       Prediction JSON files and summary CSV
results/metrics/         Training histories and validation results
src/                     Project source code
checkpoints/             Trained models (not tracked by Git)
```

Generated handcrafted features are stored in `data/features/` and are not
tracked by Git because they can be reproduced from the fixed splits.

## Environment setup

The project was tested with Python 3.13.5. Git must be installed and available
on `PATH` because the preparation script downloads the public dataset from
GitHub.

On Windows PowerShell:

```powershell
git clone https://github.com/lzk2059/final-project.git
cd final-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the computer's
CUDA environment before installing the remaining requirements. The source code
selects CUDA automatically when `torch.cuda.is_available()` is true; no
`--device` argument is required.

Check the active device with:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

All commands below must be run from the repository root with module syntax,
for example `python -m src.train`, because source files import from the `src`
package.

## 1. Download and prepare the dataset

```powershell
python -m src.prepare_data
```

No manual dataset download is required. On its first run, the script:

1. shallow-clones the public RaptorMaps InfraredSolarModules repository;
2. extracts `2020-02-14_InfraredSolarModules.zip` into
   `data/raw/extracted/`;
3. validates all referenced images and detects corrupt, duplicate, and
   conflicting samples;
4. retains the original 12 classes; and
5. creates stratified 70%/15%/15% train, validation, and test splits using
   seed 42.

The temporary repository and ZIP are removed automatically after extraction.
If a complete extracted copy already exists, it is validated and reused.

Dataset source:
<https://github.com/raptormaps/infraredsolarmodules>

## 2. Exploratory data analysis

```powershell
python -m src.eda
```

EDA outputs are written under `results/metrics/` and `results/figures/`.

## 3. Extract handcrafted features

The Random Forest baseline requires reproducible pixel-intensity, LBP, and
GLCM features:

```powershell
python -m src.features --split all
```

Use `--overwrite` to regenerate existing feature files:

```powershell
python -m src.features --split all --overwrite
```

## 4. Verify the complete experiment sequence

Print the commands and validate required inputs without training:

```powershell
python -m src.run_experiments --random-seed 42 --dry-run
```

Run a small CPU end-to-end debug test:

```powershell
python -m src.run_experiments --random-seed 42 --debug
```

Debug outputs are not valid research results.

## 5. Run formal training

Run all five deep-learning experiments and the Random Forest experiment for
one seed:

```powershell
python -m src.run_experiments --random-seed 42
```

Repeat for the other formal seeds, ideally on separate computers or GPUs:

```powershell
python -m src.run_experiments --random-seed 123
python -m src.run_experiments --random-seed 2026
```

The runner performs input checks, skips completed experiments, resumes an
interrupted deep-learning experiment from its compatible `*_last.pt`, and
automatically evaluates each best model on the validation set. Formal training
requires CUDA by default. Add `--allow-cpu` only when intentionally running the
full experiments on CPU.

One model can also be trained directly, for example:

```powershell
python -m src.train --config configs/efficientnet_multiscale_focal.yaml
```

Resume an interrupted direct run with:

```powershell
python -m src.train --config configs/efficientnet_multiscale_focal.yaml --resume checkpoints/EXPERIMENT_last.pt
```

## 6. Evaluate all formal models

After placing all 15 deep-learning `*_best.pt` files and three Random Forest
`*_best.joblib` files in `checkpoints/`, evaluate all 18 formal runs on the
held-out test split:

```powershell
python -m src.evaluate --all --split test
```

Existing complete results are skipped. To deliberately recompute and overwrite
them, add `--force`.

The command saves Accuracy, Macro-F1, Balanced Accuracy, per-class Precision,
Recall and F1, raw and normalized confusion matrices, prediction CSV files,
complexity measurements, ablation summaries, and multi-seed comparison plots.

Recreate only the comparison tables and figures from existing evaluations:

```powershell
python -m src.compare_results --split test
```

This also produces the Improved EfficientNet per-class mean F1 ± standard
deviation figure across seeds 42, 123, and 2026.

## Trained checkpoints

Checkpoints are excluded from Git because several files exceed GitHub's normal
file-size limit. The formal trained models are published in the
[`v1.0-model` GitHub Release](https://github.com/lzk2059/final-project/releases/tag/v1.0-model)
under **Assets**.

The release contains 18 best-performing checkpoints selected by validation
Macro-F1:

- 5 deep-learning experiments × 3 seeds = 15 `.pt` files;
- 1 Random Forest experiment × 3 seeds = 3 `.joblib` files.

The two additional `Source code` entries shown by GitHub are automatically
generated repository archives and do **not** contain the separately uploaded
model checkpoints.

The release tag records the source code at the time the release was created.
For the latest project code, clone or pull the `main` branch rather than using
the release's automatically generated source archive; use the release Assets
only to obtain the trained checkpoints.

Download the required checkpoint files from the release Assets list and place
them directly in:

```text
checkpoints/
```

For prediction and Grad-CAM only, the required default checkpoint is:

```text
checkpoints/efficientnet_b0_multiscale_cross_entropy_seed42_best.pt
```

It can be downloaded directly in PowerShell with:

```powershell
New-Item -ItemType Directory -Path checkpoints -Force
Invoke-WebRequest `
  -Uri "https://github.com/lzk2059/final-project/releases/download/v1.0-model/efficientnet_b0_multiscale_cross_entropy_seed42_best.pt" `
  -OutFile "checkpoints/efficientnet_b0_multiscale_cross_entropy_seed42_best.pt"
```

To reproduce `python -m src.evaluate --all --split test`, download all 18
uploaded checkpoint assets. Users with GitHub CLI installed can download them
in one step:

```powershell
New-Item -ItemType Directory -Path checkpoints -Force
gh release download v1.0-model `
  --repo lzk2059/final-project `
  --dir checkpoints `
  --pattern "*_best.pt" `
  --pattern "*_best.joblib"
```

The default deployment model is the multi-scale EfficientNet-B0 with
Cross-Entropy from seed 42, selected using validation performance. A `best`
checkpoint is used for evaluation and inference; a `last` checkpoint is needed
only to resume interrupted training.

Release page:
<https://github.com/lzk2059/final-project/releases/tag/v1.0-model>

## 7. Predict new infrared images

The inference pipeline accepts images from outside the training dataset. Place
one or more photovoltaic-module infrared images in:

```text
data/inference_images/
```

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`.
Then run:

```powershell
python -m src.inference
```

Outputs are saved in `results/inference/`:

- one JSON record per image;
- `predictions.csv` containing the predicted class, confidence, all 12 class
  probabilities, confidence status, maintenance risk, and recommendation.

The predefined maintenance-priority rules and the 0.60 review threshold are in
`configs/fault_recommendations.json`. Predictions below the threshold are
marked `Review required` and `Uncertain`. These recommendations are decision
support, not remaining-useful-life or replacement decisions.

## 8. Generate Grad-CAM explanations and case reports

After inference:

```powershell
python -m src.explain
```

Outputs are saved in `results/gradcam/` and include:

- `*_gradcam_overlay.png`: attention heatmap over the original image;
- `*_case_report.png`: original infrared image, Grad-CAM result, predicted
  class, confidence, risk level, and maintenance recommendation;
- `explanations.csv`: a summary of all explanation results.

Grad-CAM uses the exact checkpoint recorded in each successful inference JSON.

## Reproducibility and interpretation

- Training uses fixed seeds and fixed stratified splits.
- Data augmentation is applied only to the training set.
- Validation Macro-F1 selects checkpoints; the test split is reserved for
  final reporting.
- Multi-seed tables report means and sample standard deviations.
- A high softmax confidence does not guarantee correctness, especially for
  external or out-of-distribution images.
- The system predicts only the 12 trained classes and does not implement an
  unknown-class detector or true end-of-life/remaining-life estimation.

## Main output locations

```text
results/metrics/       Training and validation histories
results/evaluation/    Final metrics and per-image predictions
results/comparison/    Multi-seed and ablation summary tables
results/figures/       Curves, confusion matrices, and comparison figures
results/inference/     Text predictions and maintenance assessments
results/gradcam/       Grad-CAM overlays and complete case reports
```
