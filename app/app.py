"""
app.py

Streamlit demo: upload a structural photo, get a crack / no-crack
prediction with a confidence score. This is the piece that turns the
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
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

MODEL_PATH = Path(__file__).parent.parent / "models" / "crack_classifier.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Structural Crack Detector", page_icon="🔍")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_class


def predict(model, idx_to_class, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))

    return idx_to_class[pred_idx], float(probs[pred_idx])


st.title("🔍 Structural Crack Detector")
st.write(
    "Upload a photo of concrete, pavement, or another structural surface. "
    "The model flags whether it thinks a crack is present, as a first-pass "
    "screening tool — not a substitute for an inspector's judgment."
)

model, idx_to_class = load_model()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. "
        "Run `python src/train_classifier.py` first to produce it."
    )
else:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            label, confidence = predict(model, idx_to_class, image)
            if label == "crack":
                st.error(f"⚠️ Crack detected")
            else:
                st.success(f"✅ No crack detected")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.caption(
                "This is a screening signal, not a certified structural "
                "assessment. Low-confidence or borderline results should "
                "always go to a human reviewer."
            )

st.divider()
st.caption(
    "Portfolio project — ResNet18 transfer learning, trained on a public "
    "concrete crack dataset. See the GitHub repo README for dataset, "
    "architecture, and accuracy details."
)
