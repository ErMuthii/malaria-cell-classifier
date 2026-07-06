import numpy as np

from src.evaluate import binary_metrics, select_threshold


def test_binary_metrics_uses_parasitized_as_positive_class():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9], threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5


def test_threshold_meets_specificity_constraint_and_maximizes_sensitivity():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])

    threshold = select_threshold(y_true, probability, minimum_specificity=2 / 3)
    metrics = binary_metrics(y_true, probability, threshold)

    assert metrics["specificity"] >= 2 / 3
    assert metrics["sensitivity"] == 1.0
