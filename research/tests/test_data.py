import numpy as np

from mgc_jax import ALL_FEATURES, MFCC_FEATURES, SPECTRAL_FEATURES, load_mgc, stratified_folds
from mgc_jax.data import standardize


def test_load_shapes_and_balance():
    ds = load_mgc()
    assert ds.X.shape == (1000, 26)
    assert ds.n_classes == 10
    assert ds.features == ALL_FEATURES
    counts = np.bincount(ds.y)
    assert counts.tolist() == [100] * 10


def test_feature_subsets():
    ds = load_mgc()
    assert ds.select(MFCC_FEATURES).X.shape[1] == 20
    assert ds.select(SPECTRAL_FEATURES).X.shape[1] == 6
    assert np.allclose(ds.select(["zcr"]).X[:, 0], ds.X[:, ALL_FEATURES.index("zcr")])


def test_folds_are_stratified_and_disjoint():
    ds = load_mgc()
    folds = stratified_folds(ds.y, 10, seed=3)
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test.tolist()) == list(range(1000))
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert np.bincount(ds.y[te], minlength=10).tolist() == [10] * 10


def test_folds_reproducible():
    y = np.repeat(np.arange(10), 100)
    a = stratified_folds(y, 5, seed=7)
    b = stratified_folds(y, 5, seed=7)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))


def test_standardize_uses_train_stats_only():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(5, 2, (200, 3)), rng.normal(5, 2, (50, 3))
    Ztr, Zte = standardize(Xtr, Xte)
    assert np.allclose(Ztr.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Ztr.std(axis=0), 1, atol=1e-6)
    assert not np.allclose(Zte.mean(axis=0), 0, atol=1e-3)  # Testfold wurde nicht auf sich selbst zentriert
