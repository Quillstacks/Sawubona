import pytest

from decol.dist import Distribution


def test_from_counts_normalizes():
    d = Distribution.from_counts(["a", "b", "c"], {"a": 1, "b": 3})
    assert d.as_map["a"] == pytest.approx(0.25)
    assert d.as_map["b"] == pytest.approx(0.75)
    assert d.as_map["c"] == 0.0
    assert d.n == 4


def test_reindex_zero_fills_and_reorders():
    d = Distribution(("a", "b"), (0.5, 0.5), 10)
    r = d.reindex(["b", "a", "c"])
    assert r.categories == ("b", "a", "c")
    assert r.probs == (0.5, 0.5, 0.0)


def test_round_trip_dict():
    d = Distribution.from_counts(["no", "yes"], {"yes": 2, "no": 2})
    assert Distribution.from_dict(d.to_dict()) == d


def test_zero_mass_raises():
    with pytest.raises(ValueError):
        Distribution.from_counts(["a", "b"], {"a": 0, "b": 0})


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        Distribution(("a", "b"), (1.0,), 1)
