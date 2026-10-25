import numpy as np

from gcbench.experiments import one_world, summary
from gcbench.methods import Heads, centre, content_groups, eligible, purity
from gcbench.sim import World, authors, balanced_accuracy, new_truth, sample, train_batch


def test_centre_removes_group_means():
    x = np.arange(12, dtype=float).reshape(6, 2)
    g = np.array([0, 0, 1, 1, 1, 2])
    c = centre(x, g)
    for k in np.unique(g):
        assert np.allclose(c[g == k].mean(0), 0)


def test_authors_mixes():
    rng = np.random.default_rng(0)
    assert len(set(authors(18, 6, "distinct", rng))) == 6
    a = authors(18, 6, "dominated", rng)
    assert np.bincount(a).max() >= 3
    assert len(authors(18, 40, "repeats", rng)) == 40


def test_training_batch_is_balanced_per_topic():
    w = World(train_topics=10)
    rng = np.random.default_rng(1)
    b = train_batch(w, new_truth(w, rng), rng)
    assert len(b.y) == 10 * w.K
    for t in range(10):
        assert sorted(b.y[b.topic == t]) == list(range(w.K))


def test_content_groups_recover_clean_topics():
    w = World(noise_c=0.2)
    rng = np.random.default_rng(2)
    b = sample(w, new_truth(w, rng), 30, 6, "distinct", rng)
    g = content_groups(b.c, cut=0.5)
    assert purity(g, b.topic) > 0.99
    assert eligible(g).all()


def test_eligible_bounds():
    g = np.array([0, 1, 1] + [2] * 19)
    ok = eligible(g)
    assert not ok[0] and ok[1] and ok[2] and not ok[3:].any()


def test_twin_helps_with_true_groups_and_topic_effect():
    w = World(tau=1.5)
    rng = np.random.default_rng(3)
    t = new_truth(w, rng)
    heads = Heads(train_batch(w, t, rng))
    test = sample(w, t, 200, 6, "distinct", rng)
    plain = balanced_accuracy(test.y, heads.scores(test, None, "plain").argmax(1))
    twin = balanced_accuracy(test.y, heads.scores(test, test.topic, "twin").argmax(1))
    assert twin > plain + 0.05


def test_no_groups_leaves_plain_scores():
    w = World()
    rng = np.random.default_rng(4)
    t = new_truth(w, rng)
    heads = Heads(train_batch(w, t, rng))
    test = sample(w, t, 50, 1, "distinct", rng)
    singletons = np.arange(len(test.y))
    assert np.allclose(heads.scores(test, singletons, "twin"), heads.scores(test, None, "plain"))


def test_one_world_and_summary_are_deterministic():
    a = one_world(World(), 6, "distinct", 7)
    b = one_world(World(), 6, "distinct", 7)
    assert a == b
    s = summary([a, b], ["twin_est"])
    assert s["twin_est_se"] == 0
