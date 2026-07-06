import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from src.explain import grad_cam  # noqa: E402
from src.models import build_model  # noqa: E402


@pytest.mark.parametrize(
    ("model_name", "image_size"),
    [
        ("custom_cnn", (64, 64)),
        ("mobilenet_v2", (96, 96)),
        ("efficientnet_b0", (96, 96)),
    ],
)
def test_grad_cam_supports_all_model_architectures(model_name, image_size):
    model = build_model(model_name, image_size, weights=None)
    image = tf.random.uniform((1, *image_size, 3), maxval=255, seed=42)

    heatmap = grad_cam(model, image)

    assert heatmap.ndim == 2
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1
