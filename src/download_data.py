"""
download_data.py

Helper for getting a crack-classification dataset into the expected
data/train/{crack,no_crack} and data/val/{crack,no_crack} layout.

Run this on YOUR machine (or in Colab) — it needs real internet access,
which the assistant that generated this project does not have.

Several options are supported. Pick one.

--------------------------------------------------------------------
OPTION A (recommended, easiest): Kaggle "Concrete Crack Images for
Classification" dataset — 40,000 images, already split
positive/negative.

1. Create a free Kaggle account and API token:
   https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json
2. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600 on Mac/Linux)
3. Run:  python src/download_data.py --source kaggle

--------------------------------------------------------------------
OPTION B: SDNET2018 (56,000+ images, bridge decks/walls/pavements,
from Utah State University's data repository).

1. Go to https://digitalcommons.usu.edu/all_datasets/48/ and download
   the zip manually (no API available).
2. Unzip it somewhere, then run:
   python src/download_data.py --source sdnet --raw_dir /path/to/unzipped/SDNET2018

--------------------------------------------------------------------
OPTION C: Local dataset with Positive/ and Negative/ folders.

1. Place your dataset in a directory with Positive/ and Negative/ subfolders
   containing .jpg images.
2. Run:  python src/download_data.py --source local --raw_dir /path/to/dataset

--------------------------------------------------------------------
All options end with an 80/20 train/val split written into:
    data/train/crack/, data/train/no_crack/
    data/val/crack/,   data/val/no_crack/
which is exactly what train_classifier.py expects (torchvision ImageFolder format).
"""

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

RANDOM_SEED = 42
VAL_FRACTION = 0.2
# Original METU/Özgenel crack dataset (same images as the Kaggle mirror).
MENDELEY_ZIP_URLS = [
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip",
    "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip",
]


def split_and_copy(image_paths, label, data_root: Path, max_per_class=None):
    random.seed(RANDOM_SEED)
    image_paths = list(image_paths)
    random.shuffle(image_paths)
    if max_per_class:
        image_paths = image_paths[:max_per_class]

    n_val = int(len(image_paths) * VAL_FRACTION)
    val_paths, train_paths = image_paths[:n_val], image_paths[n_val:]

    for split, paths in [("train", train_paths), ("val", val_paths)]:
        out_dir = data_root / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, out_dir / p.name)

    print(f"  {label}: {len(train_paths)} train / {len(val_paths)} val")
    return len(train_paths), len(val_paths)


def _positive_negative_dirs(root: Path):
    positive_dir = next(root.rglob("Positive"), None)
    negative_dir = next(root.rglob("Negative"), None)
    if positive_dir is None or negative_dir is None:
        raise SystemExit(f"Could not find Positive/ and Negative/ under {root}")
    return positive_dir, negative_dir


def _split_positive_negative(root: Path, data_root: Path, max_per_class=None):
    positive_dir, negative_dir = _positive_negative_dirs(root)
    print("Splitting into train/val...")
    train_crack, val_crack = split_and_copy(positive_dir.glob("*.jpg"), "crack", data_root, max_per_class)
    train_no_crack, val_no_crack = split_and_copy(negative_dir.glob("*.jpg"), "no_crack", data_root, max_per_class)
    print(f"Created:")
    print(f"  Train crack: {train_crack}")
    print(f"  Train no_crack: {train_no_crack}")
    print(f"  Val crack: {val_crack}")
    print(f"  Val no_crack: {val_no_crack}")


def from_huggingface(data_root: Path, max_per_class=None):
    """Pull the same 40k Özgenel images from Hugging Face (no Kaggle token)."""
    from datasets import load_dataset

    print("Downloading mohammadnajeeb/concrete_crack_images from Hugging Face...")
    ds = load_dataset("mohammadnajeeb/concrete_crack_images")
    names = [n.lower() for n in ds["train"].features["label"].names]

    def dump(split, out_split, cap):
        counts = {"crack": 0, "no_crack": 0}
        for i, row in enumerate(ds[split]):
            raw = names[int(row["label"])]
            label = "crack" if "pos" in raw else "no_crack"
            if cap and counts[label] >= cap:
                if all(counts[k] >= cap for k in counts):
                    break
                continue
            out_dir = data_root / out_split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            row["image"].convert("RGB").save(out_dir / f"{label}_{i:05d}.jpg")
            counts[label] += 1
        print(f"  {out_split}: {counts}")
        return counts

    val_cap = None if not max_per_class else max(1, int(max_per_class * VAL_FRACTION))
    train_cap = None if not max_per_class else max_per_class - (val_cap or 0)
    train_counts = dump("train", "train", train_cap)
    val_split = "validation" if "validation" in ds else "test"
    val_counts = dump(val_split, "val", val_cap)

    print(f"Created:")
    print(f"  Train crack: {train_counts['crack']}")
    print(f"  Train no_crack: {train_counts['no_crack']}")
    print(f"  Val crack: {val_counts['crack']}")
    print(f"  Val no_crack: {val_counts['no_crack']}")


def from_kaggle(data_root: Path, max_per_class=None):
    import kagglehub

    print("Downloading via kagglehub (requires ~/.kaggle/kaggle.json)...")
    path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
    positive_dir, negative_dir = _positive_negative_dirs(Path(path))
    print("Splitting into train/val...")
    train_crack, val_crack = split_and_copy(positive_dir.glob("*.jpg"), "crack", data_root, max_per_class)
    train_no_crack, val_no_crack = split_and_copy(negative_dir.glob("*.jpg"), "no_crack", data_root, max_per_class)
    print(f"Created:")
    print(f"  Train crack: {train_crack}")
    print(f"  Train no_crack: {train_no_crack}")
    print(f"  Val crack: {val_crack}")
    print(f"  Val no_crack: {val_no_crack}")


def from_mendeley(data_root: Path, max_per_class=None):
    """Download the original Özgenel dataset without a Kaggle token."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "concrete-crack-images.zip"
    extract_dir = cache_dir / "mendeley_raw"

    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        last_error = None
        for url in MENDELEY_ZIP_URLS:
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"  failed: {exc}")
        if last_error:
            raise SystemExit(
                "Could not download the Mendeley zip. Place kaggle.json at "
                "~/.kaggle/kaggle.json and retry with --source kaggle.\n"
                f"Last error: {last_error}"
            )

    if not extract_dir.exists() or not any(extract_dir.rglob("*.jpg")):
        print(f"Extracting {zip_path} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    positive_dir, negative_dir = _positive_negative_dirs(extract_dir)
    print("Splitting into train/val...")
    train_crack, val_crack = split_and_copy(positive_dir.glob("*.jpg"), "crack", data_root, max_per_class)
    train_no_crack, val_no_crack = split_and_copy(negative_dir.glob("*.jpg"), "no_crack", data_root, max_per_class)
    print(f"Created:")
    print(f"  Train crack: {train_crack}")
    print(f"  Train no_crack: {train_no_crack}")
    print(f"  Val crack: {val_crack}")
    print(f"  Val no_crack: {val_no_crack}")


def from_sdnet(raw_dir: Path, data_root: Path, max_per_class=None):
    """
    SDNET2018 ships as nested folders per surface type (D=decks, P=pavements,
    W=walls), each with C (cracked) and U (uncracked) subfolders, e.g.:
        raw_dir/D/CD/*.jpg   (cracked deck)
        raw_dir/D/UD/*.jpg   (uncracked deck)
    We pool everything into two classes regardless of surface type.
    """
    crack_paths, no_crack_paths = [], []
    for jpg in raw_dir.rglob("*.jpg"):
        parent = jpg.parent.name.upper()
        if parent.startswith("C"):
            crack_paths.append(jpg)
        elif parent.startswith("U"):
            no_crack_paths.append(jpg)

    print(f"Found {len(crack_paths)} cracked / {len(no_crack_paths)} uncracked images.")
    print("Splitting into train/val...")
    train_crack, val_crack = split_and_copy(crack_paths, "crack", data_root, max_per_class)
    train_no_crack, val_no_crack = split_and_copy(no_crack_paths, "no_crack", data_root, max_per_class)
    print(f"Created:")
    print(f"  Train crack: {train_crack}")
    print(f"  Train no_crack: {train_no_crack}")
    print(f"  Val crack: {val_crack}")
    print(f"  Val no_crack: {val_no_crack}")


def from_local(raw_dir: Path, data_root: Path, max_per_class=None):
    """
    Use a local dataset with Positive/ and Negative/ folders.
    Reads from raw_dir/Positive/*.jpg and raw_dir/Negative/*.jpg
    and creates the standard train/val structure.
    """
    positive_dir = raw_dir / "Positive"
    negative_dir = raw_dir / "Negative"

    if not positive_dir.exists():
        raise SystemExit(f"Positive directory not found at {positive_dir}")
    if not negative_dir.exists():
        raise SystemExit(f"Negative directory not found at {negative_dir}")

    positive_images = list(positive_dir.glob("*.jpg"))
    negative_images = list(negative_dir.glob("*.jpg"))

    if not positive_images:
        raise SystemExit(f"No .jpg images found in {positive_dir}")
    if not negative_images:
        raise SystemExit(f"No .jpg images found in {negative_dir}")

    print(f"Found:")
    print(f"  Positive: {len(positive_images)}")
    print(f"  Negative: {len(negative_images)}")
    print("Splitting into train/val...")

    train_crack, val_crack = split_and_copy(positive_images, "crack", data_root, max_per_class)
    train_no_crack, val_no_crack = split_and_copy(negative_images, "no_crack", data_root, max_per_class)

    print(f"Created:")
    print(f"  Train crack: {train_crack}")
    print(f"  Train no_crack: {train_no_crack}")
    print(f"  Val crack: {val_crack}")
    print(f"  Val no_crack: {val_no_crack}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["kaggle", "sdnet", "mendeley", "huggingface", "local"],
                        default="huggingface")
    parser.add_argument("--raw_dir", type=str, default=None,
                         help="Required for --source sdnet and --source local: path to dataset folder")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--max_per_class", type=int, default=None,
                         help="Optional cap on images per class before the train/val split")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.source == "kaggle":
        from_kaggle(data_root, args.max_per_class)
    elif args.source == "mendeley":
        from_mendeley(data_root, args.max_per_class)
    elif args.source == "huggingface":
        from_huggingface(data_root, args.max_per_class)
    elif args.source == "local":
        if not args.raw_dir:
            raise SystemExit("--raw_dir is required for --source local")
        from_local(Path(args.raw_dir), data_root, args.max_per_class)
    else:
        if not args.raw_dir:
            raise SystemExit("--raw_dir is required for --source sdnet")
        from_sdnet(Path(args.raw_dir), data_root, args.max_per_class)

    print("\nDone. Your data/ folder is ready for train_classifier.py")
