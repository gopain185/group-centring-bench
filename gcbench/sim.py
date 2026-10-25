"""A known-truth simulation of passages whose features carry a shared topic effect.

Each world has K classes (authors) and many topics (prompts). A passage by author k on topic t has style features
x = mu[k] + u[t] + e, with u[t] ~ N(0, tau^2 I) shared by every passage on that topic and e ~ N(0, I). It also has
content features c = v[t] + noise_c * n: what the passage is about, used only to find the topic groups at test time.

The training set has every author answer every training topic once, like a balanced competition train. A test batch
has m passages per topic, with the authors inside a topic either all different, drawn with repeats, or dominated by
one author.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    K: int = 18
    d: int = 16
    sep: float = 0.75
    tau: float = 1.0
    content_dim: int = 100
    noise_c: float = 1.5
    train_topics: int = 150


@dataclass
class Truth:
    mu: np.ndarray


@dataclass
class Batch:
    x: np.ndarray
    c: np.ndarray
    y: np.ndarray
    topic: np.ndarray


def new_truth(w: World, rng: np.random.Generator) -> Truth:
    return Truth(rng.normal(0.0, w.sep, (w.K, w.d)))


def authors(K: int, m: int, mix: str, rng: np.random.Generator) -> np.ndarray:
    """The authors of the m passages on one topic."""
    if mix == "distinct":
        return rng.choice(K, min(m, K), replace=False)
    if mix == "repeats":
        return rng.integers(0, K, m)
    if mix == "dominated":  # half the passages by one author, the rest all different
        a = rng.choice(K, m, replace=False) if m <= K else rng.integers(0, K, m)
        a[: m // 2] = a[0]
        return rng.permutation(a)
    raise ValueError(mix)


def sample(w: World, t: Truth, topics: int, m: int, mix: str, rng: np.random.Generator) -> Batch:
    y = np.concatenate([authors(w.K, m, mix, rng) for _ in range(topics)])
    topic = np.repeat(np.arange(topics), len(y) // topics)
    u = rng.normal(0.0, w.tau, (topics, w.d))
    v = rng.normal(0.0, 1.0, (topics, w.content_dim))
    x = t.mu[y] + u[topic] + rng.normal(size=(len(y), w.d))
    c = v[topic] + w.noise_c * rng.normal(size=(len(y), w.content_dim))
    return Batch(x, c, y, topic)


def train_batch(w: World, t: Truth, rng: np.random.Generator) -> Batch:
    """Every author answers every training topic once."""
    return sample(w, t, w.train_topics, w.K, "distinct", rng)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean recall over the classes present in y (sklearn's definition)."""
    return float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)]))
