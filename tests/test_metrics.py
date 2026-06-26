import pytest

from decol.config import Scale
from decol.dist import Distribution
from decol.divergence import metrics

AGREE5 = Scale("agree5", "ordinal",
               ["strongly_disagree", "disagree", "neither", "agree", "strongly_agree"])
DEMO = Scale("democracy_pref", "categorical",
             ["nondemocratic_ok", "doesnt_matter", "democracy_preferable"])


def _u(scale):  # uniform
    n = len(scale.ordered)
    return Distribution(tuple(scale.ordered), tuple([1 / n] * n), 100)


def test_jsd_identical_is_zero():
    d = _u(AGREE5)
    assert metrics.jensen_shannon(d, d, AGREE5) == pytest.approx(0.0, abs=1e-9)


def test_jsd_disjoint_is_one():
    p = Distribution(tuple(AGREE5.ordered), (1, 0, 0, 0, 0), 10)
    q = Distribution(tuple(AGREE5.ordered), (0, 0, 0, 0, 1), 10)
    assert metrics.jensen_shannon(p, q, AGREE5) == pytest.approx(1.0, abs=1e-6)
    assert metrics.total_variation(p, q, AGREE5) == pytest.approx(1.0)


def test_signed_mean_shift_direction():
    low = Distribution(tuple(AGREE5.ordered), (1, 0, 0, 0, 0), 10)   # mean idx 0
    high = Distribution(tuple(AGREE5.ordered), (0, 0, 0, 0, 1), 10)  # mean idx 4
    # model high, human low -> positive shift, normalised to +1
    assert metrics.signed_mean_shift(high, low, AGREE5) == pytest.approx(1.0)
    assert metrics.signed_mean_shift(low, high, AGREE5) == pytest.approx(-1.0)


def test_ordinal_only_metrics_none_for_categorical():
    d = _u(DEMO)
    assert metrics.ordinal_mean(d, DEMO) is None
    assert metrics.signed_mean_shift(d, d, DEMO) is None
    assert metrics.wasserstein(d, d, DEMO) is None
    # but JSD still defined on unordered support
    assert metrics.jensen_shannon(d, d, DEMO) == pytest.approx(0.0, abs=1e-9)


def test_capability_gate():
    assert metrics.capability_gate(0.3, 0.4) is True
    assert metrics.capability_gate(0.5, 0.4) is False
    assert metrics.capability_gate(None, 0.4) is True   # unknown -> gated
