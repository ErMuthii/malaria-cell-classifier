"""CLI for reproducible model training."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data import load_datasets, set_global_determinism
from src.models import build_model, compile_model


def train(config_path: str | Path) -> Path:
    from tensorflow import keras

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_global_determinism(int(config["seed"]))

    splits = (config["train_split"], config["validation_split"], config["test_split"])
    train_ds, validation_ds, _ = load_datasets(
        config["image_size"], config["batch_size"], config["seed"], splits
    )
    model = build_model(
        config["model"],
        config["image_size"],
        config.get("dropout", 0.3),
        config.get("weights"),
    )
    compile_model(model, float(config["learning_rate"]))

    run_dir = Path("artifacts") / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_sensitivity", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_sensitivity",
            mode="max",
            patience=int(config["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        keras.callbacks.CSVLogger(run_dir / "epochs.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=int(config["epochs"]),
        callbacks=callbacks,
    )
    model.save(run_dir / "final.keras")
    shutil.copy2(config_path, run_dir / "config.yaml")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": __import__("tensorflow").__version__,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenet_v2.yaml")
    args = parser.parse_args()
    print(f"Best model saved to {train(args.config)}")


if __name__ == "__main__":
    main()
