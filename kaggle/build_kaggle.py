"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (two simulated test batches with their training batch, the result tables of the six sweeps
and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the sweeps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gcbench.experiments import SEED, TEST_ROWS  # noqa: E402
from gcbench.sim import World, new_truth, sample, train_batch  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/topic-grouped-passages-simulation"
NOTEBOOK_ID = "locdunhat/group-centring-when-test-topics-must-be-found"
TITLE = "Group Centring When Test Topics Must Be Found"


def _frame(name: str, split: str, b) -> pd.DataFrame:
    df = pd.concat([pd.DataFrame(b.x.round(4), columns=[f"x_{i}" for i in range(b.x.shape[1])]),
                    pd.DataFrame(b.c.round(4), columns=[f"c_{i}" for i in range(b.c.shape[1])])], axis=1)
    df.insert(0, "topic", b.topic)
    df.insert(0, "label", b.y)
    df.insert(0, "split", split)
    df.insert(0, "world", name)
    return df


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (name, m, mix) in enumerate([("m6_distinct", 6, "distinct"), ("m2_repeats", 2, "repeats")]):
        rng = np.random.default_rng([SEED, 99, i])
        w = World()
        t = new_truth(w, rng)
        parts += [_frame(name, "train", train_batch(w, t, rng)), _frame(name, "test", sample(w, t, TEST_ROWS // m, m, mix, rng))]
    pd.concat(parts).to_parquet(out / "worlds.parquet", index=False)
    for f in sorted((ROOT / "results").glob("exp*.csv")):
        (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Topic-Grouped Passages Simulation",
        "id": DATASET_ID,
        "subtitle": "Known-truth style and content features with shared topic effects",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "gcbench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/group-centring-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("group-centring-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gcbench import experiments as ex
pd.set_option("display.precision", 4); pd.set_option("display.width", 220)
results = {}
def run(fn):
    t = time.time(); df = fn(); print(fn.__name__, round(time.time() - t), "s")
    results[fn.__name__] = df
    return df.drop(columns="experiment")
"""

PLOT = """df = results["exp2_group_size"]
x = np.arange(len(df))
fig, ax = plt.subplots(figsize=(7, 4))
for col in ["twin_true", "twin_est", "score_centre_est"]:
    ax.errorbar(x, df[col], yerr=2 * df[col + "_se"], marker="o", capsize=3, label=col)
ax.set_xticks(x, df["setting"]); ax.set_xlabel("passages per topic in the test batch")
ax.set_ylabel("gain in balanced accuracy over the plain head"); ax.axhline(0, color="k", lw=0.8)
ax.legend(); plt.tight_layout(); plt.show()
"""

SAMPLE = """import glob
from gcbench.methods import Heads, content_groups, purity
from gcbench.sim import Batch, balanced_accuracy
hits = glob.glob("/kaggle/input/**/worlds.parquet", recursive=True)
if hits:
    df = pd.read_parquet(hits[0])
    xc = [c for c in df.columns if c.startswith("x_")]; cc = [c for c in df.columns if c.startswith("c_")]
    as_batch = lambda g: Batch(g[xc].to_numpy(), g[cc].to_numpy(), g.label.to_numpy(), g.topic.to_numpy())
    for world, g in df.groupby("world"):
        tr, te = as_batch(g[g.split == "train"]), as_batch(g[g.split == "test"])
        heads, est = Heads(tr), content_groups(te.c)
        row = {"plain": heads.scores(te, None, "plain"), "twin_true": heads.scores(te, te.topic, "twin"),
               "twin_est": heads.scores(te, est, "twin")}
        print(world, "purity", round(purity(est, te.topic), 3), {k: round(balanced_accuracy(te.y, v.argmax(1)), 4) for k, v in row.items()})
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. How strong the topic effect is\n\nSix passages per topic, all by different authors. "
                     "Columns are gains in balanced accuracy over the plain head (20 worlds; `*_se` is the standard "
                     "error). `twin_true` uses the true topics, `twin_est` content clusters, `score_centre_est` "
                     "subtracts each cluster's mean log-probability instead of using a second head."),
        ("code", "run(ex.exp1_topic_strength)"),
        ("markdown", "## 2. Passages per topic\n\nThe number that matters most and that you do not know for a hidden test."),
        ("code", "run(ex.exp2_group_size)"),
        ("code", PLOT),
        ("markdown", "## 3. Who wrote the passages inside a topic\n\n`distinct`: all different authors (like a "
                     "balanced train); `repeats`: authors drawn with repeats; `dominated`: half the topic by one author."),
        ("code", "run(ex.exp3_composition)"),
        ("markdown", "## 4. How cleanly the topics can be found\n\nMore noise in the content features gives less pure clusters."),
        ("code", "run(ex.exp4_purity)"),
        ("markdown", "## 5. How much weight the centred head gets"),
        ("code", "run(ex.exp5_weight)"),
        ("markdown", "## 6. The clustering cut\n\nThe same fixed cut is too loose for tight content and too strict for noisy content."),
        ("code", "run(ex.exp6_cut)"),
        ("markdown", "## 7. The worlds in the dataset\n\nTwo saved worlds (training batch plus test batch); the rules on them."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "group-centring-bench.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "group-centring-bench.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
