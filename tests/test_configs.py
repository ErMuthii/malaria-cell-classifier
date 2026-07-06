from pathlib import Path

import yaml


def test_all_training_configs_share_seed_and_non_overlapping_splits():
    configs = [yaml.safe_load(path.read_text()) for path in Path("configs").glob("*.yaml")]

    assert {config["seed"] for config in configs} == {42}
    assert {
        (config["train_split"], config["validation_split"], config["test_split"])
        for config in configs
    } == {("train[:70%]", "train[70%:85%]", "train[85%:]")}
