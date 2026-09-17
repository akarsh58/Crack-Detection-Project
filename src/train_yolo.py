"""
train_yolo.py

OPTIONAL stretch goal — Version 2 of the project. Only attempt this after
train_classifier.py is working end-to-end and you have bounding-box
labels (e.g. from Roboflow or LabelImg).

This does NOT run on the same data/ folder as the classifier — YOLO needs
its own annotation format (one .txt file per image with normalized box
coordinates) plus a dataset yaml describing where things live.

Setup:
    1. Label a subset of images with bounding boxes in Roboflow
       (roboflow.com) or LabelImg. Roboflow can export directly in
       "YOLOv8" format, which saves you from hand-writing the yaml/labels.
    2. Point --data at the exported dataset.yaml.
    3. pip install ultralytics
    4. python src/train_yolo.py --data path/to/crack_dataset.yaml

Example crack_dataset.yaml (Roboflow generates this for you):

    train: ../train/images
    val: ../valid/images
    nc: 1
    names: ['crack']
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                         help="Path to dataset.yaml (from Roboflow export or hand-built)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base_model", type=str, default="yolov8n.pt",
                         help="yolov8n.pt (fast) / yolov8s.pt (more accurate, slower)")
    args = parser.parse_args()

    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="models/yolo_runs",
        name="crack_detection",
    )

    metrics = model.val()
    print(f"\nValidation mAP50: {metrics.box.map50:.3f}")
    print("Trained weights saved under models/yolo_runs/crack_detection/weights/best.pt")
    print("Point the Streamlit app at that .pt file to swap in bounding-box detection.")


if __name__ == "__main__":
    main()
