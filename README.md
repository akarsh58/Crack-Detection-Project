# Structural Crack Detector

A ResNet18 image classifier that screens concrete, pavement, bridge-deck, and
similar surface photos for visible cracks. It is an engineering/ML portfolio
prototype, not a certified structural inspection system.

## Architecture

- **Version 1:** ImageNet-initialized ResNet18 with a two-class output:
  `crack` and `no_crack`.
- **Training:** reproducible augmentation, class-weighted cross entropy,
  configurable best-checkpoint selection (`val_accuracy`, `val_f1`, or
  `val_recall`), learning-rate reduction, and early stopping.
- **Evaluation:** validation metrics are produced during training. An
  independent test set can be evaluated with `evaluate_classifier.py`.
- **Demo:** Streamlit upload interface with model metadata and confidence
  warnings.

Accuracy alone is not sufficient for this use case: a missed crack can matter
more than a false alarm. For that reason, `val_recall` or `val_f1` can be used
for checkpoint selection.

## Repository structure

```text
app/app.py                    Streamlit demo
src/download_data.py          Dataset preparation
src/train_classifier.py       ResNet18 training and validation
src/evaluate_classifier.py    Independent test-set evaluation
src/train_yolo.py             Version 2 placeholder/roadmap entry point
models/                       Generated checkpoints and metrics (ignored)
data/                         Generated train/val data (ignored)
```

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux or Colab
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

The downloader supports Kaggle, Hugging Face, Mendeley, SDNET2018, and a local
directory containing `Positive/` and `Negative/` folders. Labels are normalized
to `crack` and `no_crack`; JPG, JPEG, and PNG files are supported.

For the local dataset already used by this project:

```bash
python src/download_data.py --source local --raw_dir dataset --data_root data --clean
```

The command validates images, performs a deterministic 80/20 split, reports
the actual copied counts, and writes:

```text
data/train/crack      data/train/no_crack
data/val/crack        data/val/no_crack
```

The split is image-level. If related crops or images from one physical
structure occur in both splits, validation can be overly optimistic. Use
source/group-level splitting when source identifiers exist, and reserve a
genuinely independent dataset for final testing.

## Training

```bash
python src/train_classifier.py --data_dir data --epochs 10 \
  --selection_metric val_recall
```

Smoke test:

```bash
python src/train_classifier.py --data_dir data --epochs 1 \
  --max_per_class 20 --early_stopping_patience 0
```

Generated under `models/`:

- `crack_classifier.pth` — weights plus class mapping, preprocessing,
  architecture/version, seed, epoch, and selection metric
- `metrics.json` — measured validation metrics, class counts, and confusion
  matrix-related outputs
- `training_curves.png`
- `confusion_matrix.png`

No performance numbers are claimed here until training is run on the target
dataset. Generated datasets and model files are ignored by Git.

## Independent evaluation

Create an independent test directory with this layout:

```text
test/crack/
test/no_crack/
```

Then run:

```bash
python src/evaluate_classifier.py \
  --checkpoint models/crack_classifier.pth \
  --data_dir test \
  --output_dir models/test_evaluation
```

This writes measured accuracy, crack precision/recall/F1, class counts,
confusion matrix, and `metrics.json`. A validation set is not an independent
test set and should not be reported as one.

## Streamlit demo

```bash
streamlit run app/app.py
```

The app accepts JPG/JPEG/PNG uploads, displays the image, predicted class,
softmax score, checkpoint version, epoch, and selection metric. The score is
explicitly presented as an uncalibrated model confidence signal, not a
probability or professional inspection conclusion. Missing, corrupt, or
incompatible checkpoints are reported in the UI.

## Limitations and safety

- The current task is image-level classification; it does not locate cracks or
  measure length, width, or severity.
- Dataset bias, image-level leakage, domain shift, and camera/lighting changes
  can materially affect results.
- A low-confidence prediction, or any result used in a real inspection, must
  be reviewed by a qualified person.
- The repository does not include datasets, secrets, or generated checkpoints.

## Version 2 roadmap

After Version 1 is stable and independently evaluated:

1. Add bounding-box labels and train YOLO for crack localization.
2. Add pixel masks and segmentation for crack extent.
3. Estimate crack length/width with calibrated image geometry.
4. Add severity categories with domain-reviewed labels and external testing.

`src/train_yolo.py` remains an optional entry point for a labeled YOLO-format
dataset; it is not part of the Version 1 training path.
