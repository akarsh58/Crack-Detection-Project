"""Evaluate a saved classifier checkpoint on an independent ImageFolder set.

Usage:
    python src/evaluate_classifier.py --checkpoint models/crack_classifier.pth \
        --data_dir test --output_dir models/test_evaluation
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from train_classifier import (
    SafeImageFolder,
    evaluate,
    get_val_transform,
    plot_confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing crack/ and no_crack/ folders")
    parser.add_argument("--output_dir", default="models/test_evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    for class_name in ("crack", "no_crack"):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise SystemExit(f"Required evaluation directory not found: {class_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    dataset = SafeImageFolder(data_dir, transform=get_val_transform())
    if dataset.class_to_idx != class_to_idx:
        raise SystemExit(
            f"Evaluation mapping {dataset.class_to_idx} does not match checkpoint "
            f"mapping {class_to_idx}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    accuracy, precision, recall, f1, cm = evaluate(
        model, loader, device, class_to_idx["crack"]
    )
    metrics = {
        "checkpoint": str(checkpoint_path),
        "images": len(dataset),
        "class_counts": {
            name: sum(1 for _, target in dataset.samples if target == index)
            for name, index in class_to_idx.items()
        },
        "accuracy": round(accuracy, 2),
        "precision_crack": round(precision, 3),
        "recall_crack": round(recall, 3),
        "f1_crack": round(f1, 3),
        "confusion_matrix": cm.tolist(),
        "selection_metric": checkpoint.get("selection_metric"),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    plot_confusion_matrix(
        cm,
        [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])],
        output_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
