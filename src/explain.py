"""Post-hoc image explanations. Attention is not proof of medical causality."""

from __future__ import annotations

import numpy as np


def _feature_layer(model):
    """Find the final spatial layer, including a nested transfer backbone."""
    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            try:
                shape = tuple(layer.output.shape)
            except (AttributeError, ValueError):
                continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer
    raise ValueError("The model has no spatial feature layer suitable for Grad-CAM")


def grad_cam(model, image_batch, layer_name: str | None = None):
    """Generate a normalized Grad-CAM map for the parasitized probability."""
    import tensorflow as tf

    layer = model.get_layer(layer_name) if layer_name else _feature_layer(model)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    with tf.GradientTape() as tape:
        feature_maps, prediction = grad_model(image_batch, training=False)
        target = prediction[:, 0]
    gradients = tape.gradient(target, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)
    heatmap = tf.nn.relu(heatmap[0])
    maximum = tf.reduce_max(heatmap)
    return (heatmap / (maximum + tf.keras.backend.epsilon())).numpy()


def overlay_heatmap(image, heatmap, alpha: float = 0.42):
    """Blend a heatmap with a PIL image and return an RGB NumPy array."""
    import matplotlib
    from PIL import Image

    rgb = np.asarray(image.convert("RGB"))
    resized = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size)
    colored = matplotlib.colormaps["jet"](np.asarray(resized) / 255.0)[..., :3] * 255
    return np.uint8(np.clip((1 - alpha) * rgb + alpha * colored, 0, 255))


def occlusion_sensitivity(model, image_batch, patch_size: int = 24, baseline: float = 127.5):
    """Measure probability reduction when square regions are hidden."""
    baseline_probability = float(model.predict(image_batch, verbose=0)[0, 0])
    array = np.asarray(image_batch).copy()
    height, width = array.shape[1:3]
    scores = np.zeros((height, width), dtype=np.float32)
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            occluded = array.copy()
            occluded[:, top : top + patch_size, left : left + patch_size, :] = baseline
            probability = float(model.predict(occluded, verbose=0)[0, 0])
            scores[top : top + patch_size, left : left + patch_size] = max(
                baseline_probability - probability, 0.0
            )
    maximum = scores.max()
    return scores / maximum if maximum else scores
