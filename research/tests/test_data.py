import numpy as np

from mgc_jax import ALL_FEATURES, MFCC_FEATURES, SPECTRAL_FEATURES, load_mgc, stratified_folds
from mgc_jax.data import standardize


def test_load_raw_shapes_and_balance():
    ds = load_mgc(dedup=False)
    assert ds.X.shape == (1000, 26)
    assert ds.n_classes == 10
    assert ds.features == ALL_FEATURES
    assert np.bincount(ds.y).tolist() == [100] * 10
    assert ds.dropped_duplicates == () and ds.dropped_conflicts == ()


def test_dedup_removes_exact_duplicates_and_conflicts():
    ds = load_mgc()
    assert len(ds.dropped_duplicates) == 13  # 14 Duplikatgruppen, eine davon ist der Label-Konflikt
    assert set(ds.dropped_conflicts) == {"metal.00058.au", "rock.00016.au"}
    assert ds.X.shape == (1000 - 13 - 2, 26)
    assert len(ds.filenames) == ds.X.shape[0]
    assert len({tuple(row) for row in ds.X.tolist()}) == ds.X.shape[0]
    assert "pop.00067.au" in ds.filenames and "pop.00071.au" not in ds.filenames


def test_select_preserves_metadata():
    ds = load_mgc()
    sub = ds.select(MFCC_FEATURES)
    assert sub.filenames == ds.filenames and sub.dropped_conflicts == ds.dropped_conflicts


def test_feature_subsets():
    ds = load_mgc()
    assert ds.select(MFCC_FEATURES).X.shape[1] == 20
    assert ds.select(SPECTRAL_FEATURES).X.shape[1] == 6
    assert np.allclose(ds.select(["zcr"]).X[:, 0], ds.X[:, ALL_FEATURES.index("zcr")])


def test_folds_are_stratified_and_disjoint():
    ds = load_mgc()
    folds = stratified_folds(ds.y, 10, seed=3)
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test.tolist()) == list(range(len(ds.y)))
    counts = np.bincount(ds.y)
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        per_class = np.bincount(ds.y[te], minlength=10)
        assert all(abs(per_class[c] - counts[c] / 10) < 1 for c in range(10))


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
