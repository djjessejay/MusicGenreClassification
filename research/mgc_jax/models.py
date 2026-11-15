"""Klassifikatoren in reinem JAX: Softmax-Regression (sizes=[d, k]) und MLP (sizes=[d, h1, ..., k]).

Kein Flax, kein Optax: Adam ist unten in 15 Zeilen implementiert. Ein jit-kompilierter
Trainingsschritt wird über alle Folds wiederverwendet, solange die Formen gleich bleiben.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Params = list[tuple[jnp.ndarray, jnp.ndarray]]


def init_params(key: jax.Array, sizes: list[int]) -> Params:
    params = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / d_in)  # He-Init, passend zu ReLU
        params.append((jax.random.normal(sub, (d_in, d_out)) * scale, jnp.zeros(d_out)))
    return params


def forward(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for W, b in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def loss_fn(params: Params, x: jnp.ndarray, y: jnp.ndarray, l2: float) -> jnp.ndarray:
    logits = forward(params, x)
    log_p = jax.nn.log_softmax(logits)
    nll = -jnp.mean(jnp.take_along_axis(log_p, y[:, None], axis=1))
    reg = sum(jnp.sum(W * W) for W, _ in params)
    return nll + l2 * reg


def _adam_init(params: Params) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


@partial(jax.jit, static_argnames=("lr", "l2"))
def _train_step(params: Params, opt: dict, x: jnp.ndarray, y: jnp.ndarray, lr: float, l2: float):
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, l2)
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, opt["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, {"m": m, "v": v, "t": t}, loss


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sizes: list[int],
    *,
    steps: int = 600,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2: float = 1e-4,
    seed: int = 0,
) -> Params:
    """Mini-Batch-Adam auf Cross-Entropy. sizes[0] muss X.shape[1], sizes[-1] die Klassenzahl sein."""
    assert sizes[0] == X.shape[1], "sizes[0] != Feature-Dimension"
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    params = init_params(k_init, sizes)
    opt = _adam_init(params)
    Xj, yj = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.int32)
    n = len(y)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        params, opt, _ = _train_step(params, opt, Xj[idx], yj[idx], lr=lr, l2=l2)
    return params


@jax.jit
def _predict(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(forward(params, x), axis=1)


def predict(params: Params, X: np.ndarray) -> np.ndarray:
    return np.asarray(_predict(params, jnp.asarray(X, dtype=jnp.float32)))
