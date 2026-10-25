"""Decision rules that use, or ignore, the topic groups of a test batch.

All rules start from two linear discriminant heads fitted on the training batch:
- the plain head reads the features as they are;
- the centred head (the "twin") reads features minus their topic's mean, fitted with the true training topics.

At test time the groups are either the true topics or clusters found from the content features. Only groups of
`lo` to `hi` passages are used; other rows keep the plain head's scores.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .sim import Batch

LO, HI = 2, 18


def centre(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    for g in np.unique(groups):
        i = groups == g
        out[i] -= x[i].mean(0)
    return out


def content_groups(c: np.ndarray, cut: float = 0.85) -> np.ndarray:
    """Average-linkage clusters of cosine distance between content vectors, cut at `cut`."""
    if len(c) < 2:
        return np.zeros(len(c), int)
    return fcluster(linkage(pdist(c, "cosine"), method="average"), cut, criterion="distance") - 1


def eligible(groups: np.ndarray, lo: int = LO, hi: int = HI) -> np.ndarray:
    size = np.bincount(groups)[groups]
    return (size >= lo) & (size <= hi)


def purity(groups: np.ndarray, topic: np.ndarray) -> float:
    """Share of same-group passage pairs that really share a topic."""
    same = (groups[:, None] == groups[None, :]) & ~np.eye(len(groups), dtype=bool)
    return float((same & (topic[:, None] == topic[None, :])).sum() / max(same.sum(), 1))


class Heads:
    def __init__(self, train: Batch):
        self.plain = LinearDiscriminantAnalysis().fit(train.x, train.y)
        self.twin = LinearDiscriminantAnalysis().fit(centre(train.x, train.topic), train.y)

    def scores(self, test: Batch, groups: np.ndarray | None, rule: str, w: float = 1.0) -> np.ndarray:
        """Log-probabilities of one rule: plain, twin (plain + w * centred head) or score_centre."""
        lp = self.plain.predict_log_proba(test.x)
        if rule == "plain" or groups is None:
            return lp
        ok = eligible(groups)
        if not ok.any():
            return lp
        out = lp.copy()
        if rule == "twin":
            out[ok] += w * self.twin.predict_log_proba(centre(test.x[ok], groups[ok]))
        elif rule == "score_centre":  # subtract each group's mean log-probability (no second head)
            out[ok] -= w * (lp[ok] - centre(lp[ok], groups[ok]))
        else:
            raise ValueError(rule)
        return out
