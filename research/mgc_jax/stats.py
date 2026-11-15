"""Statistik zum Belegen oder Widerlegen einer Hypothese aus wiederholter k-facher CV.

paired_cv_test: Gepaarte Differenzen (Modell A minus Modell B) je Fold. Folds einer
wiederholten CV sind nicht unabhängig, darum zusätzlich der korrigierte t-Test nach
Nadeau & Bengio (2003): Varianz wird mit (1/J + n_test/n_train) statt 1/J skaliert.

bootstrap_ci: Perzentil-Bootstrap des Mittelwerts der Differenzen.

mcnemar_test: Exakter McNemar-Test auf gepoolten Out-of-Fold-Vorhersagen (Binomialtest
auf den diskordanten Paaren), unabhängig von der Fold-Korrelation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_cv_test(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(diffs, dtype=np.float64)
    J = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    naive_t, naive_p = stats.ttest_rel(d, np.zeros_like(d)) if var > 0 else (np.inf, 0.0)
    corr_var = (1.0 / J + n_test / n_train) * var
    corr_t = mean / np.sqrt(corr_var) if corr_var > 0 else np.inf
    corr_p = 2 * stats.t.sf(abs(corr_t), df=J - 1)
    return {
        "n_pairs": int(J),
        "mean_diff": float(mean),
        "std_diff": float(np.sqrt(var)),
        "t_naive": float(naive_t), "p_naive": float(naive_p),
        "t_corrected": float(corr_t), "p_corrected": float(corr_p),
    }


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=np.float64)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int(np.sum(~a_ok & b_ok))  # nur B richtig
    n10 = int(np.sum(a_ok & ~b_ok))  # nur A richtig
    n = n01 + n10
    p = float(stats.binomtest(n10, n, 0.5).pvalue) if n > 0 else 1.0
    return {"only_a_correct": n10, "only_b_correct": n01, "p_exact": p}


def verdict(test: dict, ci: tuple[float, float], alpha: float = 0.05) -> str:
    """Vorregistrierte Entscheidungsregel: belegt, wenn korrigiertes p < alpha UND CI die 0 ausschliesst
    UND die Differenz positiv ist. Widerlegt (kein Effekt in der erwarteten Richtung), wenn CI die 0
    einschliesst oder die Differenz negativ ist, bei ausreichender Power. Sonst inkonklusiv."""
    lo, hi = ci
    if test["p_corrected"] < alpha and lo > 0:
        return "BELEGT"
    if hi < 0:
        return "WIDERLEGT (Effekt in Gegenrichtung)"
    if lo <= 0 <= hi:
        return "NICHT BELEGT (CI enthält 0)"
    return "INKONKLUSIV"
