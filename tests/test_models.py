import pytest

from src.models import ensemble_probability


def test_soft_vote_is_mean_probability():
    assert ensemble_probability([0.2, 0.8]) == pytest.approx(0.5)


def test_weighted_soft_vote():
    assert ensemble_probability([0.2, 0.8], [1, 3]) == pytest.approx(0.65)


@pytest.mark.parametrize("values", [[], [-0.1, 0.5], [0.5, 1.1]])
def test_soft_vote_rejects_invalid_probabilities(values):
    with pytest.raises(ValueError):
        ensemble_probability(values)
