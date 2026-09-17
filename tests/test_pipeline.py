import tempfile
import unittest
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.download_data import get_image_files, split_and_copy
from src.train_classifier import (
    _selection_metric_value,
    evaluate,
    validate_dataset_structure,
)


class FixedModel(torch.nn.Module):
    def forward(self, images):
        # First sample is no_crack, second sample is crack.
        return torch.tensor([[2.0, 0.0], [0.0, 2.0]])[: len(images)]


class PipelineTests(unittest.TestCase):
    def test_split_is_reproducible_and_supports_image_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for index, suffix in enumerate((".jpg", ".JPEG", ".png", ".jpeg")):
                image = Image.new("RGB", (8, 8), color=(index, index, index))
                image.save(source / f"image_{index}{suffix}")

            first = root / "first"
            second = root / "second"
            split_and_copy(get_image_files(source), "crack", first)
            split_and_copy(get_image_files(source), "crack", second)
            first_files = sorted(p.name for p in (first / "train" / "crack").iterdir())
            second_files = sorted(p.name for p in (second / "train" / "crack").iterdir())
            self.assertEqual(first_files, second_files)
            self.assertEqual(len(list((first / "val" / "crack").iterdir())), 1)

    def test_dataset_structure_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for label in ("crack", "no_crack"):
                    target = root / split / label
                    target.mkdir(parents=True)
                    Image.new("RGB", (8, 8)).save(target / "sample.png")
            validate_dataset_structure(root)

    def test_metric_calculation_and_selection(self):
        images = torch.zeros(2, 3, 4, 4)
        labels = torch.tensor([0, 1])
        loader = DataLoader(list(zip(images, labels)), batch_size=2)
        metrics = evaluate(FixedModel(), loader, torch.device("cpu"), pos_label=1)
        self.assertEqual(metrics[:4], (100.0, 1.0, 1.0, 1.0))
        self.assertEqual(metrics[4].tolist(), [[1, 0], [0, 1]])
        self.assertEqual(_selection_metric_value(80, 0.5, 0.9, 0.6, "val_recall"), 0.9)

    def test_checkpoint_metadata_shape(self):
        checkpoint = {
            "model_state_dict": {},
            "class_to_idx": {"crack": 0, "no_crack": 1},
            "architecture": "resnet18",
            "version": "1.0",
            "selection_metric": "val_recall",
            "epoch": 1,
            "seed": 42,
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "image_size": (224, 224),
            },
        }
        self.assertEqual(sorted(checkpoint["class_to_idx"]), ["crack", "no_crack"])
        self.assertEqual(checkpoint["preprocessing"]["image_size"], (224, 224))


if __name__ == "__main__":
    unittest.main()
