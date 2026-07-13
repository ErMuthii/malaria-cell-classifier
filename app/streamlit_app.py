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
    "Training history",
    "Ensemble learning",
    "Explain prediction",
    "Limitations and responsible use",
)

MODEL_LABELS = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNetB0",
    "soft_voting": "Soft voting ensemble",
}

MODEL_ROLES = {
    "custom_cnn": "Baseline CNN trained from scratch",
    "mobilenet_v2": "Lightweight transfer-learning candidate",
    "efficientnet_b0": "Stronger transfer-learning candidate",
    "soft_voting": "Average of the best validated models",
}

METRIC_COLUMNS = [
    "sensitivity",
    "specificity",
    "precision",
    "f2",
    "roc_auc",
    "brier_score",
    "model_size_mb",
    "mean_inference_ms_per_image",
    "threshold",
]

st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🔬", layout="wide")

st.markdown(
    """
<style>
    :root {
        --malaria-primary: #155e75;
        --malaria-primary-soft: #e0f2fe;
        --malaria-accent: #dc2626;
        --malaria-muted: #64748b;
        --malaria-border: #e2e8f0;
        --malaria-card: #ffffff;
        --malaria-bg-soft: #f8fafc;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #082f49 0%, #0f172a 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--malaria-border);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--malaria-muted);
        font-size: 0.88rem;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 750;
    }

    .hero {
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.25rem;
        color: white;
        background:
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.45), transparent 32%),
            linear-gradient(135deg, #083344 0%, #155e75 52%, #0f766e 100%);
        box-shadow: 0 18px 45px rgba(8, 47, 73, 0.22);
    }

    .hero h1 {
        font-size: 2.6rem;
        line-height: 1.05;
        margin-bottom: 0.35rem;
        color: white;
    }

    .hero p {
        color: #e0f2fe;
        font-size: 1.05rem;
        max-width: 900px;
        margin-bottom: 0;
    }

    .section-card {
        background: var(--malaria-card);
        border: 1px solid var(--malaria-border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin: 0.75rem 0 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04);
    }

    .section-card h3 {
        margin-top: 0;
        color: #0f172a;
    }

    .callout {
        border-left: 5px solid var(--malaria-primary);
        background: var(--malaria-primary-soft);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0;
        color: #0f172a;
    }

    .risk-callout {
        border-left: 5px solid var(--malaria-accent);
        background: #fef2f2;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0;
        color: #0f172a;
    }

    .status-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.22rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: #dcfce7;
        color: #166534;
    }

    .status-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }

    .small-muted {
        color: var(--malaria-muted);
        font-size: 0.92rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.warning("Research prototype — not for diagnosis or treatment.")


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="hero">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="section-card">
    <h3>{title}</h3>
    <div>{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _callout(body: str, risk: bool = False) -> None:
    class_name = "risk-callout" if risk else "callout"
    st.markdown(f'<div class="{class_name}">{body}</div>', unsafe_allow_html=True)


def _available_models() -> list[Path]:
    return sorted((ROOT / "artifacts").glob("*/best.keras"))


def _display_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def _format_percent(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def _format_number(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _evaluation_files() -> list[Path]:
    return sorted((ROOT / "results").glob("*.json"))


def _load_evaluation_results() -> dict[str, dict]:
    results: dict[str, dict] = {}
    source_specificity: dict[str, int] = {}
    for path in _evaluation_files():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model_name = result.get("model")
        if model_name:
            specificity = 1 if path.stem == f"{model_name}_evaluation" else 0
            if model_name not in results or specificity >= source_specificity.get(model_name, 0):
                results[model_name] = result
                source_specificity[model_name] = specificity
    return results


def _comparison_dataframe() -> pd.DataFrame:
    rows = []
    evaluations = _load_evaluation_results()
    known_models = ["custom_cnn", "mobilenet_v2", "efficientnet_b0", "soft_voting"]
    for model_name in known_models:
        result = evaluations.get(model_name, {})
        row = {
            "model": model_name,
            "Model": _display_name(model_name),
            "Role": MODEL_ROLES.get(model_name, "Candidate model"),
            "Status": "evaluated" if result else "needs evaluation",
        }
        for column in METRIC_COLUMNS:
            row[column] = result.get(column, np.nan)
        rows.append(row)

    dataframe = pd.DataFrame(rows)

    comparison_path = ROOT / "results" / "model_comparison.csv"
    if comparison_path.exists():
        try:
            existing = pd.read_csv(comparison_path)
        except (OSError, pd.errors.ParserError):
            existing = pd.DataFrame()
        if not existing.empty:
            for _, source in existing.iterrows():
                model_name = str(source.get("model", "")).strip()
                if model_name in set(dataframe["model"]):
                    index = dataframe.index[dataframe["model"] == model_name][0]
                    for column in METRIC_COLUMNS:
                        if column in source and pd.notna(source[column]) and pd.isna(dataframe.at[index, column]):
                            dataframe.at[index, column] = source[column]
                    if "size_mb" in source and pd.notna(source["size_mb"]) and pd.isna(dataframe.at[index, "model_size_mb"]):
                        dataframe.at[index, "model_size_mb"] = source["size_mb"]
                    if (
                        "inference_ms_per_image" in source
                        and pd.notna(source["inference_ms_per_image"])
                        and pd.isna(dataframe.at[index, "mean_inference_ms_per_image"])
                    ):
                        dataframe.at[index, "mean_inference_ms_per_image"] = source["inference_ms_per_image"]
                    if "status" in source and pd.notna(source["status"]) and dataframe.at[index, "Status"] == "needs evaluation":
                        dataframe.at[index, "Status"] = source["status"]

    return dataframe


def _numeric_results(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = dataframe[dataframe["Status"].eq("evaluated")].copy()
    for column in columns:
        available[column] = pd.to_numeric(available[column], errors="coerce")
    return available.dropna(subset=columns, how="all")


def _plot_confusion_matrix(matrix: list[list[int]], title: str):
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    image = axis.imshow(values, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks([0, 1], labels=["Uninfected", "Parasitized"])
    axis.set_yticks([0, 1], labels=["Uninfected", "Parasitized"])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure)


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
    _hero(
        "Malaria Cell Classifier",
        "A screening-oriented research prototype for segmented blood-smear cell images, "
        "designed to compare model performance, explain predictions, and communicate limitations clearly.",
    )
    _callout(
        "<strong>Presentation focus:</strong> dataset context, model comparison, threshold-aware evaluation, "
        "Grad-CAM explanations, and responsible-use boundaries. This is not a clinical diagnostic system."
    )

    left, middle, right, fourth = st.columns(4)
    left.metric("Dataset", "27,558 cells")
    middle.metric("Primary metric", "Sensitivity")
    right.metric("Positive class", "Parasitized")
    fourth.metric("Deployment target", "Streamlit")

    _section_card(
        "Problem framing",
        """
<p>Malaria microscopy depends on identifying infected red blood cells in stained blood-smear images.
This project treats the task as binary image classification:</p>
<ul>
    <li><strong>Parasitized:</strong> visual patterns are consistent with malaria infection.</li>
    <li><strong>Uninfected:</strong> those patterns are not detected in the cell image.</li>
</ul>
<p>The model is intended as a research and screening aid. A false negative is especially important
because it means an infected cell was missed.</p>
        """
    )

    _section_card(
        "Model strategy",
        """
<p>We compare models with different complexity levels:</p>
<ul>
    <li><strong>Custom CNN:</strong> baseline model trained from scratch.</li>
    <li><strong>MobileNetV2:</strong> lightweight transfer-learning model suitable for lower-resource deployment.</li>
    <li><strong>EfficientNetB0:</strong> stronger transfer-learning model with higher capacity.</li>
    <li><strong>Soft voting ensemble:</strong> averages the strongest validated model probabilities.</li>
</ul>
<p>The final recommendation should balance sensitivity, specificity, F2 score, ROC-AUC,
calibration, model size, inference time, and explainability.</p>
        """
    )

    _section_card(
        "Evaluation principle",
        """
<p>The primary metric is <strong>sensitivity for the parasitized class</strong>, because a screening
workflow should reduce missed infected cells. Sensitivity alone is not sufficient, so specificity,
precision, F2 score, ROC-AUC, and calibration are reported alongside it.</p>
<p>The decision threshold is selected on validation data and then applied to the held-out test set.</p>
        """
    )

    st.subheader("End-to-end workflow context")
    workflow = pd.DataFrame(
        [
            {"Stage": "1", "Pipeline step": "Whole-slide or smear image", "Current project status": "Future work"},
            {"Stage": "2", "Pipeline step": "Cell detection and segmentation", "Current project status": "Assumed input"},
            {"Stage": "3", "Pipeline step": "Cell classification", "Current project status": "Implemented"},
            {"Stage": "4", "Pipeline step": "Patient-level aggregation", "Current project status": "Future work"},
            {"Stage": "5", "Pipeline step": "Human expert review", "Current project status": "Required"},
        ]
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)
    _callout(
        "<strong>Clinical boundary:</strong> deployment would require external validation, patient-level evaluation, "
        "calibrated confidence estimates, integration with cell detection, and expert oversight.",
        risk=True,
    )

elif page == "Dataset exploration":
    _hero(
        "Dataset exploration",
        "Segmented cell images from the NIH/TFDS malaria dataset, shown with class examples and split assumptions.",
    )
    st.write(
        "The NIH/TFDS dataset contains segmented cell images. The configured deterministic split "
        "is 70% training, 15% validation, and 15% test. Patient/slide separation is not verified."
    )
    split_summary = pd.DataFrame(
        [
            {"Split": "Training", "Share": "70%", "Purpose": "Model fitting"},
            {"Split": "Validation", "Share": "15%", "Purpose": "Threshold selection and model selection"},
            {"Split": "Test", "Share": "15%", "Purpose": "Final held-out evaluation"},
        ]
    )
    st.dataframe(split_summary, use_container_width=True, hide_index=True)
    images = [
        path
        for path in (ROOT / "data" / "sample").glob("*/*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if images:
        st.subheader("Sample cells")
        st.image([str(path) for path in images[:12]], caption=[path.parent.name for path in images[:12]], width=140)
    else:
        st.code("python scripts/prepare_data.py --sample-size 12")
        st.info("Run the command above to download the dataset and create a local preview.")
    _callout(
        "<strong>Data limitation:</strong> cell-level splitting may leak patient or slide characteristics across "
        "partitions. This should be discussed explicitly during the presentation.",
        risk=True,
    )

elif page == "Model comparison":
    _hero(
        "Model comparison",
        "Threshold-aware comparison of screening performance, calibration, model size, and inference speed.",
    )
    comparison = _comparison_dataframe()
    display_columns = [
        "Model",
        "Role",
        "Status",
        "sensitivity",
        "specificity",
        "precision",
        "f2",
        "roc_auc",
        "brier_score",
        "model_size_mb",
        "mean_inference_ms_per_image",
        "threshold",
    ]
    table = comparison[display_columns].rename(
        columns={
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "precision": "Precision",
            "f2": "F2",
            "roc_auc": "ROC-AUC",
            "brier_score": "Brier",
            "model_size_mb": "Size MB",
            "mean_inference_ms_per_image": "Inference ms/image",
            "threshold": "Threshold",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Sensitivity": _format_percent,
                "Specificity": _format_percent,
                "Precision": _format_percent,
                "F2": _format_number,
                "ROC-AUC": _format_number,
                "Brier": _format_number,
                "Size MB": lambda value: _format_number(value, 2),
                "Inference ms/image": lambda value: _format_number(value, 2),
                "Threshold": lambda value: _format_number(value, 3),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    numeric = _numeric_results(comparison, ["sensitivity", "specificity", "precision", "f2", "roc_auc"])
    if numeric.empty:
        st.warning(
            "No evaluated model metrics were found yet. Run evaluation for each trained model, "
            "then refresh the dashboard."
        )
        st.code(
            "\n".join(
                [
                    "python -m src.evaluate --model artifacts/custom_cnn/best.keras --output results/custom_cnn_evaluation.json",
                    "python -m src.evaluate --model artifacts/mobilenet_v2/best.keras --output results/mobilenet_v2_evaluation.json",
                    "python -m src.evaluate --model artifacts/efficientnet_b0/best.keras --output results/efficientnet_b0_evaluation.json",
                ]
            )
        )
    else:
        st.subheader("Metric comparison")
        chart_data = numeric.set_index("Model")[["sensitivity", "specificity", "precision", "f2", "roc_auc"]]
        st.bar_chart(chart_data)

        left, right = st.columns(2)
        with left:
            st.subheader("Screening trade-off")
            st.scatter_chart(
                numeric,
                x="specificity",
                y="sensitivity",
                color="Model",
                size="f2",
            )
            st.caption("Upper-right is better: high sensitivity with high specificity.")
        with right:
            st.subheader("Deployment trade-off")
            deploy = _numeric_results(comparison, ["model_size_mb", "mean_inference_ms_per_image", "sensitivity"])
            if deploy.empty:
                st.info("Model size and inference-time metrics will appear after evaluation.")
            else:
                st.scatter_chart(
                    deploy,
                    x="model_size_mb",
                    y="sensitivity",
                    color="Model",
                    size="mean_inference_ms_per_image",
                )
                st.caption("This shows whether extra model size buys enough screening benefit.")

        recommended = numeric.sort_values(
            by=["sensitivity", "specificity", "f2", "roc_auc"],
            ascending=False,
        ).iloc[0]
        st.success(
            f"Current leading evaluated model by sensitivity-first ranking: "
            f"{recommended['Model']} "
            f"({recommended['sensitivity']:.1%} sensitivity, {recommended['specificity']:.1%} specificity)."
        )
        _callout(
            "Interpretation: MobileNetV2 currently gives the strongest screening-oriented tradeoff because it "
            "has the highest sensitivity while remaining close to EfficientNetB0 on specificity and ROC-AUC, "
            "with a smaller model and faster inference."
        )

elif page == "Evaluation results":
    _hero(
        "Evaluation results",
        "Model-by-model test-set results with confusion matrices, confidence intervals, and operating thresholds.",
    )
    files = _evaluation_files()
    if not files:
        st.info("Run `python -m src.evaluate --model <path>` to generate test metrics.")
    for path in files:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model_name = result.get("model", path.stem)
        st.markdown("---")
        st.subheader(_display_name(model_name))
        overview, matrix_column = st.columns([1.2, 1])
        with overview:
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Sensitivity", _format_percent(result.get("sensitivity", np.nan)))
            metric_b.metric("Specificity", _format_percent(result.get("specificity", np.nan)))
            metric_c.metric("F2 score", _format_number(result.get("f2", np.nan)))

            metric_d, metric_e, metric_f = st.columns(3)
            metric_d.metric("ROC-AUC", _format_number(result.get("roc_auc", np.nan)))
            metric_e.metric("Brier score", _format_number(result.get("brier_score", np.nan)))
            metric_f.metric("Threshold", _format_number(result.get("threshold", np.nan)))

            intervals = result.get("confidence_intervals_95", {})
            if intervals:
                st.write(
                    "95% CI — "
                    f"sensitivity: {_format_percent(intervals.get('sensitivity', [np.nan, np.nan])[0])} to "
                    f"{_format_percent(intervals.get('sensitivity', [np.nan, np.nan])[1])}; "
                    f"specificity: {_format_percent(intervals.get('specificity', [np.nan, np.nan])[0])} to "
                    f"{_format_percent(intervals.get('specificity', [np.nan, np.nan])[1])}."
                )
            st.caption(
                "Brier score measures probability calibration; lower is better. "
                "The threshold is selected on validation data, then applied to the held-out test set."
            )
        with matrix_column:
            matrix = result.get("confusion_matrix")
            if matrix:
                _plot_confusion_matrix(matrix, "Confusion matrix")
            else:
                st.info("No confusion matrix stored for this result.")

        with st.expander("Raw evaluation JSON"):
            st.json(result)
    st.write(
        "The operating threshold is selected on validation data to maximize sensitivity while "
        "maintaining acceptable specificity, then applied once to the held-out test set."
    )

elif page == "Training history":
    _hero(
        "Training history",
        "Learning curves from training logs, used to explain convergence, validation behavior, and possible instability.",
    )
    st.write(
        "These charts come from each model's `epochs.csv` file. They help show whether the model "
        "improved steadily, overfit, or produced unstable validation behavior."
    )
    histories = sorted((ROOT / "artifacts").glob("*/epochs.csv"))
    if not histories:
        st.info("No training histories found yet. Train a model to create `artifacts/<model>/epochs.csv`.")
    for history_path in histories:
        model_name = history_path.parent.name
        history = pd.read_csv(history_path)
        st.markdown("---")
        st.subheader(_display_name(model_name))
        if "epoch" not in history.columns:
            history = history.reset_index().rename(columns={"index": "epoch"})

        left, right = st.columns(2)
        with left:
            loss_columns = [column for column in ["loss", "val_loss"] if column in history]
            if loss_columns:
                st.line_chart(history.set_index("epoch")[loss_columns])
                st.caption("Training and validation loss should generally decrease.")
        with right:
            sensitivity_columns = [column for column in ["sensitivity", "val_sensitivity"] if column in history]
            if sensitivity_columns:
                st.line_chart(history.set_index("epoch")[sensitivity_columns])
                st.caption("Sensitivity measures how many parasitized cells were correctly detected.")

        left, right = st.columns(2)
        with left:
            auc_columns = [column for column in ["roc_auc", "val_roc_auc"] if column in history]
            if auc_columns:
                st.line_chart(history.set_index("epoch")[auc_columns])
        with right:
            fp_columns = [column for column in ["false_positives", "val_false_positives"] if column in history]
            if fp_columns:
                st.line_chart(history.set_index("epoch")[fp_columns])
                st.caption("False positives matter because excessive alerts reduce screening usefulness.")

elif page == "Upload cell images":
    _hero(
        "Upload cell images",
        "Batch prediction demo for segmented cell images, including model agreement and threshold-based decisions.",
    )
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
        predictions = pd.DataFrame(rows)
        predicted_positive = int((predictions["prediction"] == "Parasitized").sum())
        predicted_negative = int((predictions["prediction"] == "Uninfected").sum())
        agreement_rate = float(predictions["members_agree"].mean())
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Predicted parasitized", predicted_positive)
        metric_b.metric("Predicted uninfected", predicted_negative)
        metric_c.metric("Model agreement", f"{agreement_rate:.1%}")
        st.dataframe(
            predictions.style.format({"parasitized_probability": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download prediction CSV",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name="malaria_cell_predictions.csv",
            mime="text/csv",
        )
        _callout(
            "Probability is model confidence, not a clinical diagnosis. Batch rows represent cells, not patients."
        )

elif page == "Explain prediction":
    _hero(
        "Explain prediction",
        "Grad-CAM visual explanation for a single segmented cell image and selected model.",
    )
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
        prediction = "Parasitized" if probability >= threshold else "Uninfected"
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Parasitized probability", f"{probability:.1%}")
        metric_b.metric("Decision threshold", f"{threshold:.2f}")
        metric_c.metric("Prediction", prediction)
        _callout(
            "Grad-CAM shows where the model focused. It does not identify a medically causal feature "
            "and may expose reliance on stain, border, lighting, or acquisition artifacts.",
            risk=True,
        )

else:
    _hero(
        "Limitations and responsible use",
        "A clear boundary between a research prototype and a clinically deployable malaria diagnostic system.",
    )
    _callout("This prototype must not diagnose malaria or determine treatment.", risk=True)
    _section_card(
        "Known limitations",
        """
<ul>
    <li>Evaluation is cell-level rather than patient-level.</li>
    <li>Inputs are already cropped and segmented; real workflow performance is unknown.</li>
    <li>Patient/slide leakage has not yet been ruled out.</li>
    <li>The model has not been externally validated on Kenyan samples.</li>
    <li>Stain, lighting, microscope, and camera shifts may cause failure.</li>
    <li>Confidence may be poorly calibrated outside the test distribution.</li>
    <li>Clinical use requires prospective validation and expert oversight.</li>
</ul>
        """
    )
    _section_card(
        "Defensible conclusion",
        """
<p>The ensemble may support research and preliminary prioritization of suspicious segmented cells for
human review. It must not replace a trained microscopist.</p>
<p>The strongest future-work path is: whole-slide cell detection → cell classification →
patient-level aggregation → human review.</p>
        """,
    )
