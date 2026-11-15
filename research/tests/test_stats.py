import numpy as np

from mgc_jax.stats import bootstrap_ci, mcnemar_test, paired_cv_test, verdict


def test_corrected_p_is_more_conservative_than_naive():
    rng = np.random.default_rng(1)
    d = rng.normal(0.02, 0.03, 50)
    t = paired_cv_test(d, n_train=900, n_test=100)
    assert t["n_pairs"] == 50
    assert t["p_corrected"] > t["p_naive"]
    assert abs(t["t_corrected"]) < abs(t["t_naive"])
    lo, hi = t["ci95_corrected"]
    blo, bhi = bootstrap_ci(d)
    assert hi - lo > bhi - blo  # korrigiertes CI ist breiter als das unkorrigierte Bootstrap-CI


def test_zero_effect_not_significant():
    rng = np.random.default_rng(2)
    d = rng.normal(0.0, 0.03, 50)
    t = paired_cv_test(d, 900, 100)
    lo, hi = t["ci95_corrected"]
    assert lo < 0 < hi
    assert verdict(t).startswith("NICHT BELEGT")


def test_strong_effect_is_confirmed():
    d = np.full(50, 0.15) + np.random.default_rng(3).normal(0, 0.01, 50)
    assert verdict(paired_cv_test(d, 900, 100)) == "BELEGT"


def test_negative_effect_is_refuted():
    d = np.full(50, -0.15) + np.random.default_rng(4).normal(0, 0.01, 50)
    assert verdict(paired_cv_test(d, 900, 100)).startswith("WIDERLEGT")


def test_refutation_requires_corrected_significance():
    # Review-Finding: 25 x -0.03 und 25 x +0.01 -> Bootstrap-CI < 0, korrigierter t nur ca. -1.37
    d = np.array([-0.03] * 25 + [0.01] * 25)
    t = paired_cv_test(d, 900, 100)
    assert bootstrap_ci(d)[1] < 0
    assert t["p_corrected"] > 0.05
    assert verdict(t).startswith("NICHT BELEGT")


def test_zero_variance_cases():
    z = paired_cv_test(np.zeros(50), 900, 100)
    assert z["t_corrected"] == 0.0 and z["p_corrected"] == 1.0
    assert verdict(z).startswith("NICHT BELEGT")
    neg = paired_cv_test(np.full(50, -0.02), 900, 100)
    assert neg["t_corrected"] == -np.inf and neg["mean_diff"] < 0
    assert verdict(neg).startswith("WIDERLEGT")
    pos = paired_cv_test(np.full(50, 0.02), 900, 100)
    assert pos["t_corrected"] == np.inf and verdict(pos) == "BELEGT"


def test_mcnemar_counts_and_symmetry():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([0, 0, 0, 1, 1, 1, 0, 0])  # 5 richtig
    b = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # 5 richtig
    r = mcnemar_test(y, a, b)
    assert r["only_a_correct"] == 2 and r["only_b_correct"] == 2
    assert r["p_exact"] == 1.0
    identical = mcnemar_test(y, a, a)
    assert identical["only_a_correct"] == 0 and identical["p_exact"] == 1.0
