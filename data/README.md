# Dataset

The project uses the NIH malaria cell-image dataset exposed as `malaria` by TensorFlow Datasets (27,558 segmented RGB cell images with two balanced classes).

Run `python scripts/prepare_data.py --sample-size 12` to cache the dataset and create a small, local preview. Raw data, previews, and TFDS caches are ignored by Git because the dataset is large. Team members should use the same versioned split expressions from the YAML configs.

Important limitation: a deterministic cell-level split does **not** prove that cells from the same patient or slide are separated. Treat patient/slide leakage as unresolved unless source identifiers are obtained and group-aware splitting is performed.

Expected local layout:

```text
data/
├── raw/          # optional original files
├── processed/    # optional prepared files
├── sample/       # a few downloaded preview images
└── tensorflow_datasets/  # shared local TFDS cache used by training
```

Do not add patient-identifiable data. Confirm the source license and attribution requirements before redistributing any images.
