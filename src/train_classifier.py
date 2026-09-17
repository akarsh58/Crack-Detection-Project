"""
train_classifier.py

Version 1 of the project: crack / no-crack binary classification via
transfer learning on a frozen ResNet18 backbone.

Usage:
    python src/train_classifier.py --data_dir data --epochs 10 --unfreeze_last_block

Expects data/train/{crack,no_crack}/*.jpg and data/val/{crack,no_crack}/*.jpg
(see download_data.py to build this layout automatically).

Outputs (into --output_dir, default "models/"):
    crack_classifier.pth   - trained weights
    training_curves.png    - loss/accuracy over epochs
    confusion_matrix.png   - val set confusion matrix
    metrics.json           - final metrics, for dropping straight into your README
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _limit_per_class(dataset: datasets.ImageFolder, max_per_class: int) -> Subset:
    counts = {i: 0 for i in range(len(dataset.classes))}
    keep = []
    for idx, (_, label) in enumerate(dataset.samples):
        if counts[label] < max_per_class:
            keep.append(idx)
            counts[label] += 1
    return Subset(dataset, keep)


def build_dataloaders(data_dir: Path, batch_size: int, max_per_class: int | None):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_data = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    class_to_idx = train_data.class_to_idx
    print(f"Class mapping: {class_to_idx}")
    if "crack" not in class_to_idx:
        raise SystemExit("Expected a 'crack' class folder under train/. Got: "
                         f"{list(class_to_idx)}")

    if max_per_class:
        train_data = _limit_per_class(train_data, max_per_class)
        val_data = _limit_per_class(val_data, max(1, max_per_class // 4))

    # Windows often hangs with worker processes; keep this at 0 there.
    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, class_to_idx


def build_model(unfreeze_last_block: bool, device):
    model = models.resnet18(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_last_block:
        # Unfreezing layer4 (the last residual block) often helps accuracy
        # plateau less than a fully-frozen backbone, at the cost of slower
        # training. Good first thing to try if accuracy stalls.
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)  # binary: crack / no_crack
    return model.to(device)


def evaluate(model, loader, device, pos_label: int):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=pos_label, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, cm


def plot_training_curves(history, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Training loss")
    ax2.plot(history["val_acc"], label="val accuracy", color="green")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy (%)"); ax2.legend(); ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, output_dir: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix (val set)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _selection_metric_value(val_acc: float, precision: float, recall: float, f1: float, metric: str) -> float:
    """Return the metric value used for best-model selection.

    Higher is always better for all supported metrics. For crack detection,
    val_recall or val_f1 is usually more meaningful than val_accuracy.
    """
    if metric == "val_accuracy":
        return val_acc
    if metric == "val_f1":
        return f1
    if metric == "val_recall":
        return recall
    raise ValueError(f"Unknown selection metric: {metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--unfreeze_last_block", action="store_true",
                         help="Also fine-tune layer4, not just the final FC layer")
    parser.add_argument("--max_per_class", type=int, default=None,
                         help="Optional cap on train images per class (val uses 1/4 of this)")
    parser.add_argument(
        "--selection_metric",
        choices=["val_accuracy", "val_f1", "val_recall"],
        default="val_accuracy",
        help="Metric used to choose the best checkpoint (higher is better)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_dir, args.batch_size, args.max_per_class
    )
    pos_label = class_to_idx["crack"]
    model = build_model(args.unfreeze_last_block, device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    history = {"train_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}
    best_selection = -1.0
    best_acc = -1.0
    best_f1 = -1.0
    start_time = time.time()
    best_path = output_dir / "crack_classifier.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc, precision, recall, f1, _ = evaluate(model, val_loader, device, pos_label)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.2f}%, "
              f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

        # Model selection using configurable metric (polarity = higher-is-better)
        selection_val = _selection_metric_value(val_acc, precision, recall, f1, args.selection_metric)
        if selection_val > best_selection:
            best_selection = selection_val
            best_acc = val_acc
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
            }, best_path)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/60:.1f} min. Best val accuracy: {best_acc:.2f}%")

    # --- Last-vs-best evaluation ---
    # Evaluate the final (last-epoch) model
    last_acc, last_precision, last_recall, last_f1, last_cm = evaluate(model, val_loader, device, pos_label)

    # Load and evaluate the best checkpoint
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_acc_final, best_precision, best_recall, best_f1, best_cm = evaluate(model, val_loader, device, pos_label)

    plot_training_curves(history, output_dir)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
    plot_confusion_matrix(best_cm, class_names, output_dir)

    metrics = {
        "best_val_accuracy": round(best_acc_final, 2),
        "precision_crack": round(best_precision, 3),
        "recall_crack": round(best_recall, 3),
        "f1_crack": round(best_f1, 3),
        "epochs": args.epochs,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "unfroze_last_block": args.unfreeze_last_block,
        "positive_class": "crack",
        # Last-vs-best: shows whether the final epoch beat the saved best checkpoint
        "last_epoch": {
            "val_accuracy": round(last_acc, 2),
            "precision_crack": round(last_precision, 3),
            "recall_crack": round(last_recall, 3),
            "f1_crack": round(last_f1, 3),
        },
        "best_epoch": {
            "val_accuracy": round(best_acc_final, 2),
            "precision_crack": round(best_precision, 3),
            "recall_crack": round(best_recall, 3),
            "f1_crack": round(best_f1, 3),
        },
        "model_selection_metric": args.selection_metric,
        "model_selection_polarity": "higher_is_better",
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, plots, and metrics.json to {output_dir}/")
    print("Drop metrics.json's contents straight into your README results table.")


if __name__ == "__main__":
    main()
