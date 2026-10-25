# group-centring-bench

When several test items share a nuisance effect (passages answering the same prompt, images from the same site,
rows from the same customer), a classifier can subtract each group's mean feature vector and read what is left with a
second head trained the same way: the centred "twin". In the Name That LLM competition the top public notebooks do
this on content clusters of the hidden test, because the test has no prompt ids. This repo asks, on a simulation where
the truth is known: how much does centring gain, and what does it cost when the groups have to be found by
clustering?

## The simulation

`gcbench/sim.py`. 18 authors and many topics. A passage by author k on topic t has style features
`x = mu[k] + u[t] + e` in 16 dimensions, with an author mean `mu[k] ~ N(0, 0.75^2)`, a topic effect `u[t] ~ N(0, tau^2)`
shared by every passage on the topic (tau = 1 unless a sweep changes it) and unit noise `e`. It also has content
features `c = v[t] + 1.5 n` in 100 dimensions, which say only what the passage is about and are used to find the
topics at test time.

The training batch is balanced like a competition train: 150 topics, every author once per topic. The test batch has
about 1,800 passages with m per topic (6 unless a sweep changes it), written by all-different authors (`distinct`),
authors drawn with repeats (`repeats`), or half by one author (`dominated`).

Rules (`gcbench/methods.py`), all built from linear discriminant heads fitted on the training batch:

| rule | what it does |
| --- | --- |
| `plain` | the plain head on the features as they are |
| `twin_true` | plain log-probabilities plus the centred head's, on features minus their **true** topic mean |
| `twin_est` | the same with the topics found by clustering the content features (cosine distance, average linkage, cut 0.85) |
| `score_centre_est` | no second head: subtract each cluster's mean log-probability from the plain head's |

Only groups of 2 to 18 passages are used; other rows keep the plain scores. Every table entry is the mean gain in
balanced accuracy over `plain` across 20 worlds; standard errors are 0.001–0.007 (all of them are in `results/`).
`python -m gcbench` reproduces every number in a few minutes on a laptop CPU.

## Results

**1. Topic strength (m = 6, distinct).** With no topic effect, centring only costs a little; with a strong one, it is
the largest effect in the table. Clustering keeps about two thirds of the gain that true topics give.

| tau | plain | twin_true | twin_est | score_centre_est |
| --- | --- | --- | --- | --- |
| 0 | 0.785 | −0.006 | −0.011 | −0.132 |
| 0.5 | 0.691 | +0.052 | +0.031 | −0.077 |
| 1 | 0.551 | +0.179 | +0.123 | +0.049 |
| 2 | 0.312 | +0.404 | +0.279 | +0.225 |

**2. Passages per topic.** The number that decides the sign, and the one you do not know for a hidden test. With one
or two passages per topic the clusters are mostly wrong (pair purity 0 and 0.12) and centring on them loses.

| m | cluster purity | twin_true | twin_est | score_centre_est |
| --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.000 | −0.047 | −0.137 |
| 2 | 0.12 | +0.068 | −0.015 | −0.106 |
| 3 | 0.28 | +0.126 | +0.020 | −0.072 |
| 6 | 0.72 | +0.181 | +0.126 | +0.046 |
| 12 | 0.91 | +0.204 | +0.183 | +0.130 |
| 18 | 0.94 | +0.214 | +0.156 | +0.115 |

At m = 18 fewer rows are used (78%), because merged clusters grow past 18 passages and are skipped.

**3. Who wrote the passages in a topic.** Centring assumes the group's mean is "topic plus an average author". When
half a topic comes from one author, the mean leans towards that author and the gain shrinks to about 40%.

| test topics | twin_true | twin_est | score_centre_est |
| --- | --- | --- | --- |
| distinct, m = 6 | +0.181 | +0.126 | +0.053 |
| repeats, m = 6 | +0.147 | +0.098 | +0.017 |
| dominated, m = 6 | +0.070 | +0.044 | −0.063 |
| distinct, m = 18 | +0.217 | +0.160 | +0.119 |
| repeats, m = 18 | +0.182 | +0.134 | +0.089 |
| dominated, m = 18 | +0.075 | +0.054 | −0.022 |

**4. How cleanly the topics can be found (m = 6).** More noise in the content view gives less pure clusters; below
a purity of about 0.1 the twin loses. Very clean content gives impure clusters too at this fixed cut, because
the cut is then too loose (see 6).

| content noise | cluster purity | twin_true | twin_est |
| --- | --- | --- | --- |
| 0.5 | 0.36 | +0.181 | +0.065 |
| 1.5 | 0.72 | +0.179 | +0.120 |
| 2.5 | 0.06 | +0.179 | −0.034 |
| 3.5 | 0.01 | +0.181 | −0.042 |

**5. Weight of the centred head (estimated topics).** A small weight is the hedge: at m = 2, where full weight
loses, 0.25 still gains a little.

| test topics | w 0.25 | w 0.5 | w 1 | w 2 |
| --- | --- | --- | --- | --- |
| distinct, m = 6 | +0.076 | +0.107 | +0.126 | +0.130 |
| distinct, m = 2 | +0.008 | +0.002 | −0.015 | −0.034 |
| repeats, m = 6 | +0.066 | +0.091 | +0.105 | +0.103 |

**6. The clustering cut.** Gain of the twin (w = 1) and, in brackets, the pair purity and the share of rows in
usable clusters, for four cuts. The best cut moves with how tight the content is. A cut that is too loose merges
topics into impure clusters and loses; a cut that is too strict leaves almost everything unclustered and simply does
nothing. With the right cut even two passages per topic gain (+0.066 at content noise 0.5).

| content noise, m | cut 0.5 | cut 0.7 | cut 0.85 | cut 0.95 |
| --- | --- | --- | --- | --- |
| 0.5, m = 2 | **+0.066** (1.00, 1.00) | +0.054 (0.68, 1.00) | +0.009 (0.18, 1.00) | +0.002 (0.04, 0.30) |
| 1.5, m = 2 | +0.001 (0.99, 0.01) | −0.007 (0.47, 0.76) | −0.015 (0.12, 1.00) | −0.001 (0.03, 0.61) |
| 2.5, m = 2 | 0.000 (0.05, 0.00) | −0.058 (0.03, 0.68) | −0.039 (0.01, 1.00) | −0.004 (0.00, 0.65) |
| 0.5, m = 6 | **+0.181** (1.00, 1.00) | +0.177 (0.96, 1.00) | +0.064 (0.37, 0.91) | +0.001 (0.11, 0.01) |
| 1.5, m = 6 | +0.003 (1.00, 0.05) | +0.092 (0.94, 0.88) | **+0.124** (0.73, 1.00) | +0.016 (0.18, 0.29) |
| 2.5, m = 6 | 0.000 (0.10, 0.00) | −0.047 (0.13, 0.70) | −0.034 (0.06, 1.00) | −0.002 (0.02, 0.63) |

## What to take from it

- Centring inside groups removes a shared nuisance exactly when the groups are right, and the twin head is better than
  subtracting mean log-probabilities: score centring also removes the authors' own signal and loses whenever the
  topic effect is weak or the groups are small.
- With estimated groups the sign depends on how many passages per topic the test has and how pure the clusters are.
  Both are unknown for a hidden test, so check the cluster sizes you get on the test (singletons, pairs, 4–18) and
  prefer a small twin weight unless the clusters look like full prompts.
- The cut is the setting to get right, and its safe side is the strict one: a strict cut that groups little costs
  nothing, while a loose cut that merges topics loses. Look at the purity you would get on a labelled set with the
  same structure before choosing it.
- Centring assumes a group holds a mix of authors. A test where one author writes much of a topic breaks that
  assumption and cuts the gain to about 40%.

## Check on real out-of-fold predictions

Name That LLM, prompt-grouped 5-fold OOF of a three-member blend (Qwen3-8B two-stage LoRA 1.0, DeBERTa-v3-large 0.5,
n-gram SVC 1.0). Every prompt is cut to m answers, topics are found by content TF-IDF clustering inside each fold, and
a share w of the chosen members' mean log-probability is subtracted in clusters of 2 to 18 (score centring, since
these members have no twin head). Balanced accuracy, 5 subsamples per m:

| centred members, weight | m = 18 | m = 6 | m = 3 | m = 2 |
| --- | --- | --- | --- | --- |
| none | 0.626 | 0.620 | 0.620 | 0.634 |
| DeBERTa, 0.75 | 0.639 | 0.631 | 0.624 | 0.637 |
| LoRA, 0.25 | 0.633 | 0.627 | 0.627 | 0.639 |
| LoRA, 1.0 | 0.628 | 0.620 | 0.592 | 0.588 |
| all three, 0.25 | 0.644 | 0.633 | 0.624 | 0.640 |
| all three, 0.5 | 0.646 | 0.636 | 0.616 | 0.613 |

The real data behave like table 2 and 5 together: light centring gains at every m, heavy centring gains with full
prompts and loses with two or three answers per prompt. The weights were picked on these rows, so the best cells are
slightly optimistic.

## Limits

The topic effect is a pure additive shift in a Gaussian feature space, which centring removes exactly; real topic
effects are only partly additive. The heads are linear discriminant analysis. Content is independent of the author.

## Layout

- `gcbench/sim.py`: worlds, sampling, balanced accuracy
- `gcbench/methods.py`: the two heads, centring, content clustering, purity
- `gcbench/experiments.py`, `python -m gcbench`: the sweeps; results in `results/exp*.csv`
- `tests/`: `python -m pytest`
- `kaggle/`: builder for the Kaggle dataset and notebook

MIT licence.
