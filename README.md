# Structural Crack Detector

A computer vision project that detects cracks/defects in structural photos
(concrete, pavement, bridge decks), wrapped in a clickable web demo.

**Problem statement:** manual visual inspection of structures is slow and
doesn't scale. This model pre-screens photos so an engineer only reviews
the ones flagged as likely damaged.

**Status:** 🚧 scaffolded, not yet trained — see "Your next steps" below.

---

## Project structure

```
crack-detection-project/
├── data/                    # train/val images (you populate this — see step 2)
│   ├── train/{crack,no_crack}/
│   └── val/{crack,no_crack}/
├── src/
│   ├── download_data.py     # pulls a public dataset into data/
│   ├── train_classifier.py  # Version 1: crack/no-crack classifier (ResNet18)
│   └── train_yolo.py        # Version 2 (optional): bounding-box detection
├── app/
│   └── app.py                # Streamlit demo app
├── models/                  # trained weights + metrics land here (gitignored)
└── requirements.txt
```

---

## Your next steps, in order

### Step 1 — Environment setup (~10 min)

```bash
cd crack-detection-project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If your laptop has no GPU, do steps 3-4 in **Google Colab** instead (free
GPU, and you can `!pip install -r requirements.txt` there too) — training
ResNet on CPU for thousands of images will be painfully slow.

### Step 2 — Get data (~15-30 min)

Pick ONE:

- **Easiest:** Kaggle's "Concrete Crack Images for Classification"
  (40k images, pre-split positive/negative).
  ```bash
  python src/download_data.py --source kaggle
  ```
  (Needs a free Kaggle API token — instructions are in the script's docstring.)

- **Use local dataset:** If you have a local dataset with Positive/ and Negative/ folders:
  ```bash
  python src/download_data.py --source local --raw_dir dataset
  ```
  This reads from `dataset/Positive/*.jpg` and `dataset/Negative/*.jpg` and creates
  the standard train/val structure with an 80/20 split.

- **More data, real bridges:** SDNET2018 (56k+ images). Manual download
  required, then:
  ```bash
  python src/download_data.py --source sdnet --raw_dir /path/to/SDNET2018
  ```

- **Most impressive for interviews:** do either of the above, then ALSO
  drop 200-500 of your own site/inspection photos into
  `data/train/crack/`, `data/train/no_crack/`, etc. (check you have rights
  to use them if they're from work). This is what makes the project
  yours instead of a dataset tutorial — and gives you a real story to
  tell in an interview.

Either path leaves you with `data/train/{crack,no_crack}` and
`data/val/{crack,no_crack}` full of images, which is what step 3 expects.

**Note:** The local dataset is not committed to GitHub to avoid repository bloat.

### Step 3 — Train the classifier (~20-60 min depending on hardware)

```bash
python src/train_classifier.py --data_dir data --epochs 10
```

This fine-tunes a ResNet18 pretrained on ImageNet, freezing everything
except the final layer (fast, works well with a few thousand images).

For a quick smoke test to verify the pipeline works:
```bash
python src/train_classifier.py --data_dir data --epochs 2 --max_per_class 500
```

Outputs land in `models/`:
- `crack_classifier.pth` — the trained weights the app loads
- `training_curves.png` — loss/accuracy over epochs, for your README
- `confusion_matrix.png` — for your README
- `metrics.json` — final numbers, copy straight into the results table below

**If accuracy plateaus:**
```bash
python src/train_classifier.py --data_dir data --epochs 15 --unfreeze_last_block
```
This also fine-tunes the last ResNet block instead of just the final layer.
Other fixes: more augmentation (already includes flip/rotation/color
jitter), or check for class imbalance in your data.

### Step 4 — Run the demo app (~5 min)

```bash
streamlit run app/app.py
```

Opens at `localhost:8501`. Upload a test image, see the prediction and
confidence score live.

### Step 5 — Deploy it (~10 min)

Push this repo to GitHub, then deploy free at
[streamlit.io/cloud](https://streamlit.io/cloud), pointing it at
`app/app.py`. You'll get a public URL — link it directly in your
resume/LinkedIn. A live clickable demo beats a GitHub link alone.

**Note:** `models/crack_classifier.pth` needs to be in the repo (or
pulled at startup) for the deployed app to have a model to load — if
it's large, consider Git LFS or downloading it from a release asset at
app startup instead of committing it directly.

### Step 6 (optional stretch) — Upgrade to bounding-box detection

Once classification works, label a subset of images with bounding boxes
in [Roboflow](https://roboflow.com) (free tier is enough), export in
YOLOv8 format, then:

```bash
python src/train_yolo.py --data path/to/exported/dataset.yaml
```

See `src/train_yolo.py`'s docstring for the full walkthrough. This locates
*where* a crack is, not just whether one exists — more visually
impressive, more work.

---

## Results

*(Fill this in from `models/metrics.json` after training — this table is
what an interviewer actually looks at.)*

| Version | Dataset | Images | Val Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| v1 (ResNet18, frozen) | Kaggle Concrete Crack | — | — | — | — | — |

---

## The one paragraph for interviews

*(Fill in after training, e.g.:)* "I built a crack-detection model using
transfer learning on ResNet18, trained on [dataset]. It's wrapped in a
Streamlit app where you upload a structural photo and get a flagged
prediction with model confidence score. The real-world angle: this could
pre-screen thousands of drone or site-inspection photos and surface only
the ones an engineer actually needs to look at closely, cutting manual
review time significantly."

---

## Notes

- Start with classification even if you eventually want detection — much
  shorter feedback loop, gets you a working demo fast.
- Keep every experiment (model version, accuracy, dataset size) in the
  results table above as you iterate — shows methodical work.
- The confidence score in the app matters more than a bare yes/no for a
  real inspection workflow — low-confidence predictions should route to
  a human, not get auto-accepted.
- **Data leakage considerations:** The current random 80/20 split is acceptable
  for a first prototype, but note that random image-level splitting can lead
  to overly optimistic validation results if related/source images or patches
  from the same structure appear in both train and val sets. For better
  generalization testing, consider:
  - External evaluation on another dataset such as SDNET2018
  - Source-level splitting when source/group information is available
  - The current approach is suitable for prototyping but not for production
  deployment without additional validation on truly independent data.
