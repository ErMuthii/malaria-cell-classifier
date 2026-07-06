"""Presentation dashboard for the malaria classifier research prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit executes this file with app/ as the import root. Add the repository
# root so the local src package works both locally and on Community Cloud.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from src.data import prepare_image
from src.explain import grad_cam, overlay_heatmap
from src.models import ensemble_probability

PAGES = (
    "Project overview",
    "Dataset exploration",
    "Model comparison",
    "Evaluation results",
    "Upload cell images",
    "Explain prediction",
    "Limitations and responsible use",
)

st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🔬", layout="wide")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.warning("Research prototype — not for diagnosis or treatment.")


def _available_models() -> list[Path]:
    return sorted((ROOT / "artifacts").glob("*/best.keras"))


@st.cache_resource
def _load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def _model_config(path: Path) -> dict:
    config_path = path.parent / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shape = _load_model(str(path)).input_shape[1:3]
    return {"model": path.parent.name, "image_size": list(shape)}


def _predict(path: Path, image: Image.Image) -> float:
    config = _model_config(path)
    batch = prepare_image(image, config["image_size"])
    return float(_load_model(str(path)).predict(batch, verbose=0)[0, 0])


def _model_picker(multiselect: bool = False):
    paths = _available_models()
    if not paths:
        st.info(
            "No trained models are installed. Train a model or place a release artifact at "
            "`artifacts/<model>/best.keras`. The rest of the dashboard remains available."
        )
        return [] if multiselect else None
    labels = {path.parent.name: path for path in paths}
    if multiselect:
        chosen = st.multiselect("Ensemble members", labels, default=list(labels)[:2], max_selections=2)
        return [labels[name] for name in chosen]
    name = st.selectbox("Model", labels)
    return labels[name]


if page == "Project overview":
    st.title("Malaria Cell Classifier")
    st.subheader("Evidence-based screening research, with human review")
    st.write(
        "We compare a custom CNN baseline, MobileNetV2, EfficientNetB0, and soft voting. "
        "The selected model must balance sensitivity, specificity, F2, calibration, model size, "
        "inference time, and explainability—not merely maximize accuracy."
    )
    left, middle, right = st.columns(3)
    left.metric("Dataset", "27,558 cells")
    middle.metric("Primary metric", "Sensitivity")
    right.metric("Positive class", "Parasitized")
    st.markdown(
        "**Proposed workflow:** whole-slide cell detection → cell classification → "
        "patient-level aggregation → trained microscopist review."
    )

elif page == "Dataset exploration":
    st.title("Dataset exploration")
    st.write(
        "The NIH/TFDS dataset contains segmented cell images. The configured deterministic split "
        "is 70% training, 15% validation, and 15% test. Patient/slide separation is not verified."
    )
    images = [
        path
        for path in (ROOT / "data" / "sample").glob("*/*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if images:
        st.image([str(path) for path in images[:12]], caption=[path.parent.name for path in images[:12]], width=140)
    else:
        st.code("python scripts/prepare_data.py --sample-size 12")
        st.info("Run the command above to download the dataset and create a local preview.")
    st.warning("Cell-level splitting may leak patient or slide characteristics across partitions.")

elif page == "Model comparison":
    st.title("Model comparison")
    comparison_path = ROOT / "results" / "model_comparison.csv"
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    ["Custom CNN", "Baseline", "Pending", "Pending", "Pending", "Pending"],
                    ["MobileNetV2", "Lightweight transfer", "Pending", "Pending", "Pending", "Pending"],
                    ["EfficientNetB0", "Stronger transfer", "Pending", "Pending", "Pending", "Pending"],
                    ["Soft voting", "Best two candidates", "Pending", "Pending", "Pending", "Pending"],
                ],
                columns=["Model", "Role", "Sensitivity", "Specificity", "Size (MB)", "Inference (ms)"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Pending values are deliberate; do not present metrics before reproducible evaluation.")

elif page == "Evaluation results":
    st.title("Evaluation results")
    files = sorted((ROOT / "results").glob("*.json"))
    if not files:
        st.info("Run `python -m src.evaluate --model <path>` to generate test metrics.")
    for path in files:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        st.subheader(result.get("model", path.stem))
        st.json(result)
    st.write(
        "The operating threshold is selected on validation data to maximize sensitivity while "
        "maintaining acceptable specificity, then applied once to the held-out test set."
    )

elif page == "Upload cell images":
    st.title("Upload one or more segmented cell images")
    selected_models = _model_picker(multiselect=True)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    uploads = st.file_uploader(
        "PNG or JPEG cell images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if uploads and selected_models:
        rows = []
        for upload in uploads:
            image = Image.open(upload).convert("RGB")
            probabilities = {path.parent.name: _predict(path, image) for path in selected_models}
            probability = ensemble_probability(list(probabilities.values()))
            rows.append(
                {
                    "file": upload.name,
                    "prediction": "Parasitized" if probability >= threshold else "Uninfected",
                    "parasitized_probability": probability,
                    "members_agree": len({value >= threshold for value in probabilities.values()}) == 1,
                    **probabilities,
                }
            )
        st.dataframe(pd.DataFrame(rows).style.format({"parasitized_probability": "{:.1%}"}), use_container_width=True)
        st.caption(
            "Probability is model confidence, not a clinical diagnosis. Batch rows represent cells, not patients."
        )

elif page == "Explain prediction":
    st.title("Explain a prediction with Grad-CAM")
    selected_model = _model_picker()
    upload = st.file_uploader("Cell image", type=["png", "jpg", "jpeg"], key="explain")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="explain_threshold")
    if upload and selected_model:
        image = Image.open(upload).convert("RGB")
        config = _model_config(selected_model)
        batch = prepare_image(image, config["image_size"])
        model = _load_model(str(selected_model))
        probability = float(model.predict(batch, verbose=0)[0, 0])
        heatmap = grad_cam(model, batch)
        original, explanation = st.columns(2)
        original.image(image, caption="Uploaded cell", use_column_width=True)
        explanation.image(overlay_heatmap(image, heatmap), caption="Grad-CAM overlay", use_column_width=True)
        st.metric("Parasitized probability", f"{probability:.1%}")
        st.write(f"Prediction at threshold {threshold:.2f}: **{'Parasitized' if probability >= threshold else 'Uninfected'}**")
        st.warning(
            "Grad-CAM shows where the model focused. It does not identify a medically causal feature "
            "and may expose reliance on stain, border, lighting, or acquisition artifacts."
        )

else:
    st.title("Limitations and responsible use")
    st.error("This prototype must not diagnose malaria or determine treatment.")
    st.markdown(
        """
- Evaluation is cell-level rather than patient-level.
- Inputs are already cropped and segmented; real workflow performance is unknown.
- Patient/slide leakage has not yet been ruled out.
- The model has not been externally validated on Kenyan samples.
- Stain, lighting, microscope, and camera shifts may cause failure.
- Confidence may be poorly calibrated outside the test distribution.
- Clinical use requires prospective validation and expert oversight.
        """
    )
    st.write(
        "The ensemble may support research and preliminary prioritization of suspicious segmented "
        "cells for human review. It must not replace a trained microscopist."
    )
