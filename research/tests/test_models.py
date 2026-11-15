import jax.numpy as jnp
import numpy as np

from mgc_jax import load_mgc, predict, stratified_folds, train_classifier
from mgc_jax.data import standardize
from mgc_jax.models import init_params, loss_fn


def _fold0():
    ds = load_mgc()
    tr, te = stratified_folds(ds.y, 5, seed=0)[0]
    Xtr, Xte = standardize(ds.X[tr], ds.X[te])
    return Xtr, ds.y[tr], Xte, ds.y[te]


def test_training_reduces_loss():
    Xtr, ytr, _, _ = _fold0()
    sizes = [26, 10]
    p0 = init_params(__import__("jax").random.PRNGKey(0), sizes)
    before = float(loss_fn(p0, jnp.asarray(Xtr), jnp.asarray(ytr), 1e-4))
    p1 = train_classifier(Xtr, ytr, sizes, steps=200, lr=5e-3)
    after = float(loss_fn(p1, jnp.asarray(Xtr), jnp.asarray(ytr), 1e-4))
    assert after < before * 0.7


def test_softmax_and_mlp_beat_chance():
    Xtr, ytr, Xte, yte = _fold0()
    for sizes in ([26, 10], [26, 64, 10]):
        p = train_classifier(Xtr, ytr, sizes, steps=300, lr=3e-3)
        acc = float(np.mean(predict(p, Xte) == yte))
        assert acc > 0.3, f"{sizes}: acc={acc}"


def test_training_is_deterministic():
    Xtr, ytr, Xte, _ = _fold0()
    a = predict(train_classifier(Xtr, ytr, [26, 32, 10], steps=50, seed=5), Xte)
    b = predict(train_classifier(Xtr, ytr, [26, 32, 10], steps=50, seed=5), Xte)
    assert np.array_equal(a, b)


def test_size_mismatch_raises():
    Xtr, ytr, _, _ = _fold0()
    try:
        train_classifier(Xtr, ytr, [20, 10], steps=1)
    except AssertionError:
        return
    raise AssertionError("erwartete AssertionError bei falscher Eingangsdimension")
