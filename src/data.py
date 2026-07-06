"""Deterministic input pipelines for the TFDS malaria dataset."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

# Model convention after remapping the TFDS labels: 0=uninfected, 1=parasitized.
CLASS_NAMES = ("uninfected", "parasitized")
DEFAULT_SPLITS = ("train[:70%]", "train[70%:85%]", "train[85%:]")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tensorflow_datasets"


def set_global_determinism(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow and request deterministic kernels."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _resize(image, label, image_size: tuple[int, int]):
    import tensorflow as tf

    image = tf.image.resize(tf.cast(image, tf.float32), image_size, antialias=True)
    # TFDS malaria uses 0=parasitized and 1=uninfected. Make infection positive.
    return image, 1.0 - tf.cast(label, tf.float32)


def load_datasets(
    image_size: Iterable[int] = (160, 160),
    batch_size: int = 32,
    seed: int = 42,
    splits: tuple[str, str, str] = DEFAULT_SPLITS,
    data_dir: str | Path | None = None,
):
    """Return deterministic train, validation, and test `tf.data` pipelines."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    size = tuple(int(value) for value in image_size)
    datasets = tfds.load(
        "malaria",
        split=list(splits),
        as_supervised=True,
        data_dir=str(data_dir or DEFAULT_DATA_DIR),
        shuffle_files=False,
    )

    prepared = []
    for index, dataset in enumerate(datasets):
        dataset = dataset.map(
            lambda image, label: _resize(image, label, size),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        if index == 0:
            dataset = dataset.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        prepared.append(dataset)
    return tuple(prepared)


def prepare_image(image, image_size: Iterable[int]):
    """Convert a PIL/NumPy RGB image into a float32 model batch."""
    import tensorflow as tf

    tensor = tf.convert_to_tensor(np.asarray(image.convert("RGB")), dtype=tf.float32)
    tensor = tf.image.resize(tensor, tuple(image_size), antialias=True)
    return tf.expand_dims(tensor, axis=0)
