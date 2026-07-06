"""Post-hoc image explanations. Attention is not proof of medical causality."""

from __future__ import annotations

import numpy as np


def grad_cam(model, image_batch, layer_name: str | None = None):
    """Generate a normalized Grad-CAM map for the parasitized probability."""
    import tensorflow as tf

    requested_layer_found = layer_name is None
    feature_maps = None

    # Walking the outer model eagerly keeps nested application backbones connected
    # to the classifier graph. Accessing a nested model's symbolic `layer.output`
    # directly is disconnected from the outer input in Keras 3.
    with tf.GradientTape() as tape:
        value = tf.convert_to_tensor(image_batch)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                value = layer(value, training=False)
            except TypeError:
                value = layer(value)

            is_spatial = getattr(value.shape, "rank", len(value.shape)) == 4
            is_requested = layer_name is None or layer.name == layer_name
            if is_spatial and is_requested:
                feature_maps = value
                tape.watch(feature_maps)
                requested_layer_found = True

        prediction = value
        target = prediction[:, 0]

    if not requested_layer_found:
        raise ValueError(f"Layer '{layer_name}' was not found in the outer model")
    if feature_maps is None:
        raise ValueError("The model has no spatial feature layer suitable for Grad-CAM")

    gradients = tape.gradient(target, feature_maps)
    if gradients is None:
        raise ValueError("The selected feature layer is not connected to the model prediction")
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
