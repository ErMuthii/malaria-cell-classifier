# Malaria Cell Classifier

Research prototype for classifying segmented red-blood-cell images as `parasitized` or `uninfected`. The project compares a custom CNN, MobileNetV2, EfficientNetB0, and a soft-voting ensemble, with sensitivity as the primary selection metric.

> **Not a medical device.** This software is for education and research. It must not diagnose malaria, determine treatment, or replace a trained microscopist.

## Reproducible setup

The supported Python version is **3.11** on Windows and Apple Silicon macOS. TensorFlow runs on CPU on both platforms; native-Windows GPU training is not supported by current TensorFlow releases.

```bash
git clone git@github.com:ErMuthii/malaria-cell-classifier.git
cd malaria-cell-classifier
python3.11 -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install and verify:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

## Data

The 300+ MB dataset is never committed. Download it through TensorFlow Datasets:

```bash
python scripts/prepare_data.py --sample-size 12
```

This caches the full dataset outside Git and writes a small local preview to `data/sample/`. See [data/README.md](data/README.md) for storage and leakage cautions.

## Train and evaluate

Each run saves its exact configuration, history, and model artifact. Start with the fast baseline, then train the transfer-learning candidates:

```bash
python -m src.train --config configs/custom_cnn.yaml
python -m src.train --config configs/mobilenet_v2.yaml
python -m src.train --config configs/efficientnet_b0.yaml

python -m src.evaluate --model artifacts/mobilenet_v2/best.keras
python -m src.evaluate --model artifacts/efficientnet_b0/best.keras
```

Model selection must balance sensitivity, specificity, F2, calibration, size, inference time, and explanation quality—not accuracy alone. Evaluation is currently cell-level and does not establish patient-level performance.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes project context, data exploration, model comparison, evaluation, single/batch upload, Grad-CAM, and responsible-use limitations.

For Streamlit Community Cloud:

1. Choose `app/streamlit_app.py` as the entrypoint.
2. Open **Advanced settings** and set **Python version** to **3.11** before deploying or rebooting the app. TensorFlow `2.16.2` does not provide wheels for newer runtimes such as Python 3.14.
3. Upload trained `.keras` files using a GitHub Release or cloud storage; do not commit them to Git.

## Team workflow

Create a branch per task, open a pull request, and require another member to review before merging into `main`. Before implementation begins, each member should reproduce setup and tests on a separate computer.

```bash
git switch -c feature/short-description
git push -u origin feature/short-description
```

## Repository map

- `configs/`: versioned training parameters and seeds
- `src/`: data, architectures, training, evaluation, and explanations
- `app/`: Streamlit presentation dashboard
- `tests/`: fast checks that do not download the full dataset
- `artifacts/`: local model files (ignored except documentation)
- `results/`: small metrics/tables suitable for Git
- `notebooks/`: exploration only; reusable logic belongs in `src/`

## Intended next steps

Whole-slide cell detection → cell classification → patient-level aggregation → human review. Before any clinical use, the system requires external Kenyan validation, leakage-safe patient/slide splits, prospective evaluation, calibration monitoring, and expert oversight.
