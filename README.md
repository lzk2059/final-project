# Final Project

Machine-learning project structure for data preparation, training, evaluation, explanation, and inference.

## Layout

- `data/`: raw, processed, and split datasets
- `configs/`: experiment configurations
- `src/`: project source code
- `results/`: figures, metrics, predictions, and EOL cases
- `checkpoints/`: saved model checkpoints

## Dataset preparation

Run `python src/prepare_data.py`. On the first run, the script shallow-clones
the official InfraredSolarModules GitHub repository into a temporary directory,
extracts the dataset into `data/raw/extracted/`, and automatically removes the
temporary repository and ZIP file. Later runs reuse the extracted dataset and
do not download it again. Git must be installed and available on `PATH`.
