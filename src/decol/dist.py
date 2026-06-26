"""A categorical distribution over an item's ordered scale categories.

Shared by the data layer (human baselines) and the divergence metrics (model answer
distributions) so the two are always compared on identical, ordered support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Distribution:
    categories: tuple[str, ...]   # ordered support (matches the item's scale.ordered)
    probs: tuple[float, ...]      # normalised, same length as categories
    n: float                      # (effective) sample size behind the estimate

    def __post_init__(self) -> None:
        if len(self.categories) != len(self.probs):
            raise ValueError("categories and probs must be the same length")

    @property
    def as_map(self) -> dict[str, float]:
        return dict(zip(self.categories, self.probs))

    def reindex(self, categories: Sequence[str]) -> "Distribution":
        """Return a copy on a (possibly reordered/extended) support, zero-filling."""
        m = self.as_map
        probs = tuple(float(m.get(c, 0.0)) for c in categories)
        return Distribution(tuple(categories), probs, self.n)

    def to_dict(self) -> dict:
        return {"categories": list(self.categories),
                "probs": list(self.probs), "n": self.n}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Distribution":
        return cls(tuple(d["categories"]), tuple(float(p) for p in d["probs"]),
                   float(d["n"]))

    @classmethod
    def from_counts(cls, categories: Sequence[str],
                    counts: Mapping[str, float] | Iterable[float]) -> "Distribution":
        """Build a normalised distribution from per-category counts/weights.

        ``counts`` may be a {category: weight} mapping or a sequence aligned to
        ``categories``. Categories with no mass are kept at probability 0.
        """
        categories = tuple(categories)
        if isinstance(counts, Mapping):
            vec = [float(counts.get(c, 0.0)) for c in categories]
        else:
            vec = [float(x) for x in counts]
            if len(vec) != len(categories):
                raise ValueError("counts length must match categories")
        total = sum(vec)
        if total <= 0:
            raise ValueError("cannot build a distribution with zero total mass")
        return cls(categories, tuple(v / total for v in vec), total)
