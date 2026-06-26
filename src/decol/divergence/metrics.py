"""Scale-native divergence metrics between a model's answer distribution and the human
marginal, computed directly on each Afrobarometer item's own categories.

Primary measure: **Jensen-Shannon distance** (symmetric, bounded [0, 1], base-2).
Direction: **signed ordinal mean-shift** (model mean - human mean), so we can say e.g.
"on press freedom the model sits more pro-control than isiZulu speakers". For ordinal
scales we also expose 1-D Wasserstein (earth-mover) distance, which respects category
order. No libertarian/authoritarian projection is imposed.
"""
from __future__ import annotations

import numpy as np

from ..config import Scale
from ..dist import Distribution


def _aligned(p: Distribution, q: Distribution, scale: Scale) -> tuple[np.ndarray, np.ndarray]:
    cats = scale.ordered
    pv = np.asarray(p.reindex(cats).probs, dtype=float)
    qv = np.asarray(q.reindex(cats).probs, dtype=float)
    return pv, qv


def jensen_shannon(p: Distribution, q: Distribution, scale: Scale) -> float:
    """JS distance in [0, 1] (base-2; sqrt of the JS divergence)."""
    from scipy.spatial.distance import jensenshannon
    pv, qv = _aligned(p, q, scale)
    d = float(jensenshannon(pv, qv, base=2))
    return 0.0 if np.isnan(d) else d


def total_variation(p: Distribution, q: Distribution, scale: Scale) -> float:
    pv, qv = _aligned(p, q, scale)
    return float(0.5 * np.abs(pv - qv).sum())


def ordinal_mean(d: Distribution, scale: Scale) -> float | None:
    """Mean category index (0..n-1) weighted by probability; None for unordered scales."""
    if not scale.is_ordinal:
        return None
    v = np.asarray(d.reindex(scale.ordered).probs, dtype=float)
    idx = np.arange(len(scale.ordered), dtype=float)
    return float((v * idx).sum())


def signed_mean_shift(model: Distribution, human: Distribution, scale: Scale,
                      normalized: bool = True) -> float | None:
    """model_mean - human_mean on the ordinal scale. Positive = model sits higher on the
    scale (toward ``scale.ordered[-1]``). Normalised to [-1, 1] by (n-1) when requested."""
    mm, hm = ordinal_mean(model, scale), ordinal_mean(human, scale)
    if mm is None or hm is None:
        return None
    shift = mm - hm
    if normalized and len(scale.ordered) > 1:
        shift /= (len(scale.ordered) - 1)
    return float(shift)


def wasserstein(model: Distribution, human: Distribution, scale: Scale) -> float | None:
    if not scale.is_ordinal:
        return None
    from scipy.stats import wasserstein_distance
    mv, hv = _aligned(model, human, scale)
    pos = np.arange(len(scale.ordered), dtype=float)
    return float(wasserstein_distance(pos, pos, mv, hv))


def item_divergence(model: Distribution, human: Distribution, scale: Scale) -> dict:
    """All metrics for one (model, human) pair on one item's scale."""
    return {
        "jsd": jensen_shannon(model, human, scale),
        "tv": total_variation(model, human, scale),
        "wasserstein": wasserstein(model, human, scale),
        "mean_shift": signed_mean_shift(model, human, scale),
    }


def capability_gate(capability: float | None, tau: float) -> bool:
    """True when capability is below threshold τ -> 'low-capability regime' (a finding,
    not a forced score). Unknown capability (None) is treated as below threshold."""
    if capability is None:
        return True
    return capability < tau
