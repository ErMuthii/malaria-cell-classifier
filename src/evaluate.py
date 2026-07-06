"""Threshold-aware evaluation for binary malaria cell classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    roc_auc_score,
)

from src.data import DEFAULT_SPLITS, load_datasets


def select_threshold(y_true, y_probability, minimum_specificity: float = 0.80) -> float:
    """Maximize sensitivity among thresholds satisfying minimum specificity."""
    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    candidates = np.unique(np.r_[0.0, y_probability, 1.0])
    eligible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if specificity >= minimum_specificity:
            eligible.append((sensitivity, specificity, float(threshold)))
    if not eligible:
        return 0.5
    # Prefer sensitivity, then specificity, then a threshold near 0.5.
    return max(eligible, key=lambda row: (row[0], row[1], -abs(row[2] - 0.5)))[2]


def binary_metrics(y_true, y_probability, threshold: float = 0.5) -> dict[str, float | list]:
    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_intervals(y_true, y_probability, threshold: float, iterations: int = 1000, seed: int = 42):
    """Return percentile 95% CIs for sensitivity and specificity."""
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)
    rng = np.random.default_rng(seed)
    samples = {"sensitivity": [], "specificity": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[indices]).size < 2:
            continue
        metrics = binary_metrics(y_true[indices], y_probability[indices], threshold)
        for key in samples:
            samples[key].append(metrics[key])
    return {
        key: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for key, values in samples.items()
    }


def evaluate_model(model_path: str | Path, config_path: str | Path | None = None) -> dict:
    import tensorflow as tf
    import yaml

    model_path = Path(model_path)
    if config_path is None:
        config_path = model_path.parent / "config.yaml"
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    splits = (config["train_split"], config["validation_split"], config["test_split"])
    _, validation_ds, test_ds = load_datasets(
        config["image_size"], config["batch_size"], config["seed"], splits
    )
    model = tf.keras.models.load_model(model_path)

    validation_y = np.concatenate([labels.numpy() for _, labels in validation_ds])
    validation_probability = model.predict(validation_ds, verbose=0).ravel()
    threshold = select_threshold(validation_y, validation_probability)

    test_y = np.concatenate([labels.numpy() for _, labels in test_ds])
    start = time.perf_counter()
    test_probability = model.predict(test_ds, verbose=0).ravel()
    elapsed = time.perf_counter() - start
    metrics = binary_metrics(test_y, test_probability, threshold)
    metrics["confidence_intervals_95"] = bootstrap_intervals(test_y, test_probability, threshold)
    metrics["model_size_mb"] = model_path.stat().st_size / (1024**2)
    metrics["mean_inference_ms_per_image"] = elapsed * 1000 / len(test_y)
    metrics["model"] = config["model"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    metrics = evaluate_model(args.model, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
