"""Model factories for baseline and transfer-learning candidates."""

from __future__ import annotations

import os
from collections.abc import Sequence


def _configure_certificate_bundle() -> None:
    """Give Python/Keras a portable CA bundle for HTTPS weight downloads."""
    if "SSL_CERT_FILE" not in os.environ:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()


def _augmentation():
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.10),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_custom_cnn(image_size: Sequence[int], dropout: float = 0.3):
    from tensorflow import keras

    inputs = keras.Input((*image_size, 3), name="image")
    x = _augmentation()(inputs)
    x = keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def _build_transfer(model_name: str, image_size: Sequence[int], dropout: float, weights: str | None):
    from tensorflow import keras

    _configure_certificate_bundle()
    builders = {
        "mobilenet_v2": keras.applications.MobileNetV2,
        "efficientnet_b0": keras.applications.EfficientNetB0,
    }
    builder = builders[model_name]
    backbone = builder(include_top=False, weights=weights, input_shape=(*image_size, 3))
    backbone.trainable = False

    inputs = keras.Input((*image_size, 3), name="image")
    x = _augmentation()(inputs)
    # EfficientNet includes its own rescaling; MobileNetV2 expects [-1, 1].
    if model_name == "mobilenet_v2":
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="preprocess")(x)
    else:
        x = keras.layers.Activation("linear", name="preprocess")(x)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    return keras.Model(inputs, outputs, name=model_name)


def build_model(
    model_name: str,
    image_size: Sequence[int],
    dropout: float = 0.3,
    weights: str | None = "imagenet",
):
    """Build one of the three supported binary classifiers."""
    if model_name == "custom_cnn":
        return build_custom_cnn(image_size, dropout)
    if model_name in {"mobilenet_v2", "efficientnet_b0"}:
        return _build_transfer(model_name, image_size, dropout, weights)
    raise ValueError(f"Unknown model '{model_name}'")


def compile_model(model, learning_rate: float = 3e-4):
    import tensorflow as tf
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Recall(name="sensitivity"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            tf.keras.metrics.FalsePositives(name="false_positives"),
            tf.keras.metrics.TrueNegatives(name="true_negatives"),
        ],
    )
    return model


def ensemble_probability(probabilities: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Return a validated soft-voting probability."""
    import numpy as np

    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be a non-empty sequence within [0, 1]")
    if weights is None:
        return float(values.mean())
    vote_weights = np.asarray(weights, dtype=float)
    if vote_weights.shape != values.shape or np.any(vote_weights < 0) or vote_weights.sum() <= 0:
        raise ValueError("Weights must match probabilities and have a positive sum")
    return float(np.average(values, weights=vote_weights))
