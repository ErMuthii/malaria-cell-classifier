"""Download TFDS malaria and export a tiny local preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image


def prepare(sample_size: int, output_dir: Path) -> None:
    dataset, info = tfds.load("malaria", split="train", with_info=True, as_supervised=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class = max(1, sample_size // 2)
    counts = {0: 0, 1: 0}
    names = info.features["label"].names
    for image, label in tfds.as_numpy(dataset):
        index = int(label)
        if counts[index] >= per_class:
            continue
        destination = output_dir / names[index]
        destination.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(destination / f"sample_{counts[index]:02d}.png")
        counts[index] += 1
        if all(count >= per_class for count in counts.values()):
            break
    print(f"Dataset: {info.splits['train'].num_examples:,} images")
    print(f"Preview: {sum(counts.values())} images written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    prepare(args.sample_size, args.output_dir)


if __name__ == "__main__":
    main()
