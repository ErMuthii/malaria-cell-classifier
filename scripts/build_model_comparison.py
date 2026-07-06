"""Build the dashboard model-comparison table from evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_ROLES = {
    "custom_cnn": "Baseline CNN trained from scratch",
    "mobilenet_v2": "Lightweight transfer-learning candidate",
    "efficientnet_b0": "Stronger transfer-learning candidate",
    "soft_voting": "Average of the best validated models",
}


def build_comparison(results_dir: Path) -> pd.DataFrame:
    results: dict[str, tuple[int, dict]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        model = result.get("model")
        if not model:
            continue
        specificity = 1 if path.stem == f"{model}_evaluation" else 0
        if model not in results or specificity >= results[model][0]:
            results[model] = (specificity, result)

    rows = [
        {
            "model": model,
            "role": MODEL_ROLES.get(model, "Candidate model"),
            "sensitivity": result.get("sensitivity"),
            "specificity": result.get("specificity"),
            "precision": result.get("precision"),
            "f2": result.get("f2"),
            "roc_auc": result.get("roc_auc"),
            "brier_score": result.get("brier_score"),
            "size_mb": result.get("model_size_mb"),
            "inference_ms_per_image": result.get("mean_inference_ms_per_image"),
            "threshold": result.get("threshold"),
            "status": "evaluated",
        }
        for model, (_, result) in results.items()
    ]
    return pd.DataFrame(rows).sort_values(["sensitivity", "specificity"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/model_comparison.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_comparison(results_dir).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
