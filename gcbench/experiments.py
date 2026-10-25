"""The sweeps behind the README tables. Every row averages `WORLDS` independent worlds.

Each world: draw class means, fit both heads on a balanced training batch (150 topics x 18 authors), then score a
test batch of about 1,800 passages with m passages per topic.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .methods import Heads, content_groups, eligible, purity
from .sim import World, balanced_accuracy, new_truth, sample, train_batch

SEED = 20261026
WORLDS = 20
TEST_ROWS = 1800

RULES = [  # (column, rule, grouping, weight)
    ("twin_true", "twin", "true", 1.0),
    ("twin_est", "twin", "est", 1.0),
    ("score_centre_est", "score_centre", "est", 1.0),
]


def one_world(w: World, m: int, mix: str, seed: int, rules=RULES) -> dict:
    rng = np.random.default_rng(seed)
    truth = new_truth(w, rng)
    heads = Heads(train_batch(w, truth, rng))
    test = sample(w, truth, max(1, TEST_ROWS // m), m, mix, rng)
    found = {"true": test.topic, "est": content_groups(test.c)}
    row = dict(plain=balanced_accuracy(test.y, heads.scores(test, None, "plain").argmax(1)),
               purity=purity(found["est"], test.topic), grouped=float(eligible(found["est"]).mean()))
    for col, rule, grouping, wt in rules:
        if grouping not in found:  # "cut=0.6": content clusters at another cut
            found[grouping] = content_groups(test.c, float(grouping.split("=")[1]))
            row["purity_" + grouping] = purity(found[grouping], test.topic)
            row["grouped_" + grouping] = float(eligible(found[grouping]).mean())
        groups = found[grouping]
        row[col] = balanced_accuracy(test.y, heads.scores(test, groups, rule, wt).argmax(1))
    return row


def summary(rows: list[dict], cols: list[str]) -> dict:
    df = pd.DataFrame(rows)
    out = {c: df[c].mean() for c in df.columns if c in ("plain", "purity", "grouped") or c.startswith(("purity_", "grouped_"))}
    for c in cols:
        gain = df[c] - df["plain"]
        out[c] = gain.mean()
        out[c + "_se"] = gain.std(ddof=1) / np.sqrt(len(df))
    return out


def sweep(name: str, settings: list[tuple], rules=RULES) -> pd.DataFrame:
    """settings: (label, world, m, mix) tuples. Columns hold the gain in balanced accuracy over the plain head."""
    out = []
    for i, (label, w, m, mix) in enumerate(settings):
        rows = [one_world(w, m, mix, SEED + 1000 * i + j, rules) for j in range(WORLDS)]
        out.append({"setting": label, **summary(rows, [r[0] for r in rules])})
    df = pd.DataFrame(out)
    df.insert(0, "experiment", name)
    return df


def exp1_topic_strength() -> pd.DataFrame:
    return sweep("topic_strength", [(f"tau={t}", World(tau=t), 6, "distinct") for t in [0.0, 0.5, 1.0, 2.0]])


def exp2_group_size() -> pd.DataFrame:
    return sweep("group_size", [(f"m={m}", World(), m, "distinct") for m in [1, 2, 3, 6, 12, 18]])


def exp3_composition() -> pd.DataFrame:
    return sweep("composition", [(f"{mix}, m={m}", World(), m, mix)
                                 for m in [6, 18] for mix in ["distinct", "repeats", "dominated"]])


def exp4_purity() -> pd.DataFrame:
    return sweep("purity", [(f"noise_c={n}", World(noise_c=n), 6, "distinct") for n in [0.5, 1.5, 2.5, 3.5]])


def exp5_weight() -> pd.DataFrame:
    rules = [(f"twin_w{wt}", "twin", "est", wt) for wt in [0.25, 0.5, 1.0, 2.0]]
    return sweep("weight", [(f"{mix}, m={m}", World(), m, mix) for m, mix in [(6, "distinct"), (2, "distinct"),
                                                                               (6, "repeats")]], rules)


def exp6_cut() -> pd.DataFrame:
    cuts = [0.5, 0.7, 0.85, 0.95]
    rules = [(f"twin_cut{c}", "twin", f"cut={c}", 1.0) for c in cuts]
    return sweep("cut", [(f"noise_c={n}, m={m}", World(noise_c=n), m, "distinct")
                         for m in [2, 6] for n in [0.5, 1.5, 2.5]], rules)


ALL = [exp1_topic_strength, exp2_group_size, exp3_composition, exp4_purity, exp5_weight, exp6_cut]
