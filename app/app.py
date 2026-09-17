"""
app.py

Streamlit demo: upload a structural photo, get a crack / no-crack
prediction with a model confidence score. This is the piece that turns the
project from "a notebook" into a clickable portfolio artifact.

Run locally:
    streamlit run app/app.py

Deploy free at https://streamlit.io/cloud by pointing it at your GitHub
repo + this file path.
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
import torch.nn as nn

MODEL_PATH = Path(__file__).parent.parent / "models" / "crack_classifier.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Structural Crack Detector", page_icon="🔍")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        class_to_idx = checkpoint["class_to_idx"]
        state_dict = checkpoint["model_state_dict"]
        if set(class_to_idx) != {"crack", "no_crack"}:
            raise ValueError(f"Unsupported class mapping: {class_to_idx}")
        if sorted(class_to_idx.values()) != [0, 1]:
            raise ValueError(f"Class indices must be [0, 1]: {class_to_idx}")

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
        model.load_state_dict(state_dict)
        model.eval()

        preprocessing = checkpoint.get("preprocessing", {})
        image_size = preprocessing.get("image_size", (224, 224))
        mean = preprocessing.get("mean", IMAGENET_MEAN)
        std = preprocessing.get("std", IMAGENET_STD)
        return model, idx_to_class, {
            "version": checkpoint.get("version", "1.0"),
            "architecture": checkpoint.get("architecture", "resnet18"),
            "epoch": checkpoint.get("epoch"),
            "selection_metric": checkpoint.get("selection_metric"),
            "image_size": tuple(image_size),
            "mean": mean,
            "std": std,
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, None, {"error": f"Could not load model checkpoint: {exc}"}


def predict(model, idx_to_class, image: Image.Image, model_info=None):
    try:
        model_info = model_info or {}
        transform = transforms.Compose([
            transforms.Resize(tuple(model_info.get("image_size", (224, 224)))),
            transforms.ToTensor(),
            transforms.Normalize(
                model_info.get("mean", IMAGENET_MEAN),
                model_info.get("std", IMAGENET_STD),
            ),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs))

        return idx_to_class[pred_idx], float(probs[pred_idx]), None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as e:
        return None, None, str(e)


st.title("🔍 Structural Crack Detector")
st.write(
    "Upload a photo of concrete, pavement, or another structural surface. "
    "The model flags whether it thinks a crack is present, as a first-pass "
    "screening tool — not a substitute for an inspector's judgment. "
    "The confidence score is the model's softmax output, not a calibrated probability."
)

model, idx_to_class, model_info = load_model()

if model is None:
    if model_info and model_info.get("error"):
        st.error(model_info["error"])
    else:
        st.warning(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run `python src/train_classifier.py` first to produce it."
        )
else:
    # Ensure model_info is not None (backward compatibility)
    if model_info is None:
        model_info = {"version": "unknown", "architecture": "unknown"}

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            image = Image.open(uploaded)
        except (OSError, UnidentifiedImageError) as e:
            st.error(f"Failed to load image: {e}")
            st.stop()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            label, confidence, error = predict(model, idx_to_class, image, model_info)
            if error:
                st.error(f"Prediction failed: {error}")
            else:
                if label == "crack":
                    st.error(f"⚠️ Crack detected")
                else:
                    st.success(f"✅ No crack detected")
                st.metric("Model confidence score", f"{confidence*100:.1f}%")

                # Confidence warning
                if confidence < 0.7:
                    st.warning("⚠️ Low confidence - Human review recommended")
                elif confidence < 0.85:
                    st.info("ℹ️ Moderate confidence - Consider human review")

                st.caption(
                    "This is a screening signal, not a certified structural "
                    "assessment. The confidence score is the model's softmax output, "
                    "not a calibrated probability. Low-confidence or borderline results "
                    "should always go to a human reviewer."
                )

                # Model information
                model_details = (
                    f"Model: {model_info['architecture']} v{model_info['version']}"
                )
                if model_info.get("epoch"):
                    model_details += f" | checkpoint epoch: {model_info['epoch']}"
                if model_info.get("selection_metric"):
                    model_details += f" | selected by: {model_info['selection_metric']}"
                st.caption(model_details)

st.divider()
st.caption(
    "Portfolio project — ResNet18 transfer learning, trained on a public "
    "concrete crack dataset. See the GitHub repo README for dataset, "
    "architecture, and accuracy details."
)
