"""Statistik zum Belegen oder Widerlegen einer Hypothese aus wiederholter k-facher CV.

paired_cv_test: Gepaarte Differenzen (Modell A minus Modell B) je Fold. Folds einer
wiederholten CV sind nicht unabhängig, darum der korrigierte t-Test nach Nadeau & Bengio
(2003): Varianz wird mit (1/J + n_test/n_train) statt 1/J skaliert. Das zurückgegebene
Konfidenzintervall ci95_corrected nutzt dieselbe korrigierte Varianz und ist die Grundlage
des Urteils. Null-Varianz wird explizit behandelt: keine Differenz heisst t=0, p=1.

bootstrap_ci: Perzentil-Bootstrap des Mittelwerts. Behandelt Folds als unabhängig und ist
darum zu schmal; nur zur Information, nicht für das Urteil.

mcnemar_test: Exakter McNemar-Test auf gepoolten Out-of-Fold-Vorhersagen (Binomialtest
auf den diskordanten Paaren), unabhängig von der Fold-Korrelation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_cv_test(diffs: np.ndarray, n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    d = np.asarray(diffs, dtype=np.float64)
    J = len(d)
    if J < 2:
        raise ValueError("mindestens 2 gepaarte Differenzen nötig")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    corr_var = (1.0 / J + n_test / n_train) * var
    if var == 0.0:
        # Konstante Differenzen: ohne Streuung ist kein Test möglich. Keine Differenz -> nicht signifikant.
        if mean == 0.0:
            t_naive = t_corr = 0.0
            p_naive = p_corr = 1.0
        else:
            t_naive = t_corr = float(np.sign(mean)) * np.inf
            p_naive = p_corr = 0.0
        ci = (mean, mean)
    else:
        t_naive = mean / np.sqrt(var / J)
        p_naive = 2 * stats.t.sf(abs(t_naive), df=J - 1)
        t_corr = mean / np.sqrt(corr_var)
        p_corr = 2 * stats.t.sf(abs(t_corr), df=J - 1)
        half = stats.t.ppf(1 - alpha / 2, df=J - 1) * np.sqrt(corr_var)
        ci = (mean - half, mean + half)
    return {
        "n_pairs": int(J),
        "mean_diff": mean,
        "std_diff": float(np.sqrt(var)),
        "t_naive": float(t_naive), "p_naive": float(p_naive),
        "t_corrected": float(t_corr), "p_corrected": float(p_corr),
        "ci95_corrected": (float(ci[0]), float(ci[1])),
        "alpha": alpha,
    }


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Unkorrigiertes Perzentil-Bootstrap-CI. Nur informativ, siehe Modul-Docstring."""
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


def verdict(test: dict) -> str:
    """Vorregistrierte, symmetrische Entscheidungsregel auf Basis des korrigierten Tests:
    BELEGT, wenn p_korrigiert < alpha und das korrigierte CI vollständig > 0 liegt.
    WIDERLEGT, wenn p_korrigiert < alpha und das korrigierte CI vollständig < 0 liegt.
    Sonst NICHT BELEGT: die Daten reichen für keine Richtung aus."""
    lo, hi = test["ci95_corrected"]
    significant = test["p_corrected"] < test["alpha"]
    if significant and lo > 0:
        return "BELEGT"
    if significant and hi < 0:
        return "WIDERLEGT (Effekt in Gegenrichtung)"
    return "NICHT BELEGT (kein signifikanter Effekt nach Fold-Korrektur)"
