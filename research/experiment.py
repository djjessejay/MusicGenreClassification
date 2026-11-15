#!/usr/bin/env python3
"""Experten-Skill: Hypothese -> Experiment -> Belegen/Widerlegen -> Dokumentieren.

Führt wiederholte stratifizierte k-fache CV für alle Konfigurationen aus, wertet die
vorregistrierten Hypothesen (hypotheses/*.md) statistisch aus und schreibt
results/results.json sowie results/REPORT.md.

Beispiel:
    python experiment.py                      # Vollauf: 5 x 10-fold, 600 Schritte
    python experiment.py --quick              # Rauchtest: 1 x 3-fold, 60 Schritte
    python experiment.py --repeats 10 --out results_r10
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np

from mgc_jax import ALL_FEATURES, MFCC_FEATURES, load_mgc, predict, stratified_folds, train_classifier
from mgc_jax.data import standardize
from mgc_jax.stats import bootstrap_ci, mcnemar_test, paired_cv_test, verdict

# --------------------------------------------------------------------------------------
# Konfigurationen (Name -> Featureset + Architektur + Optimierer). sizes[0] wird gesetzt.
# --------------------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "softmax_all":  {"features": ALL_FEATURES,  "hidden": [],        "lr": 5e-3, "l2": 1e-4},
    "mlp_all":      {"features": ALL_FEATURES,  "hidden": [128, 64], "lr": 1e-3, "l2": 1e-4},
    "softmax_mfcc": {"features": MFCC_FEATURES, "hidden": [],        "lr": 5e-3, "l2": 1e-4},
    "mlp_mfcc":     {"features": MFCC_FEATURES, "hidden": [128, 64], "lr": 1e-3, "l2": 1e-4},
}

# Vorregistrierte Hypothesen. Behauptung ist immer: accuracy(a) > accuracy(b).
HYPOTHESES: list[dict] = [
    {"id": "H1", "a": "mlp_all", "b": "softmax_all",
     "title": "Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell auf den 26 Features"},
    {"id": "H2", "a": "mlp_all", "b": "mlp_mfcc",
     "title": "Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (MLP)"},
    {"id": "H3", "a": "softmax_all", "b": "softmax_mfcc",
     "title": "Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (Softmax)"},
]


def run_cv(ds, *, repeats: int, folds: int, steps: int, batch_size: int, log=print) -> dict:
    n_cfg = len(CONFIGS)
    acc = {name: np.zeros((repeats, folds)) for name in CONFIGS}
    oof = {name: np.zeros((repeats, len(ds.y)), dtype=np.int64) for name in CONFIGS}
    t0 = time.time()
    for r in range(repeats):
        for k, (tr, te) in enumerate(stratified_folds(ds.y, folds, seed=1000 + r)):
            for name, cfg in CONFIGS.items():
                sub = ds.select(cfg["features"])
                Xtr, Xte = standardize(sub.X[tr], sub.X[te])
                sizes = [Xtr.shape[1], *cfg["hidden"], ds.n_classes]
                params = train_classifier(Xtr, ds.y[tr], sizes, steps=steps, batch_size=batch_size,
                                          lr=cfg["lr"], l2=cfg["l2"], seed=r * folds + k)
                pred = predict(params, Xte)
                acc[name][r, k] = float(np.mean(pred == ds.y[te]))
                oof[name][r, te] = pred
            done = (r * folds + k + 1) * n_cfg
            log(f"  repeat {r+1}/{repeats} fold {k+1}/{folds}  "
                f"{'  '.join(f'{n}={acc[n][r,k]:.3f}' for n in CONFIGS)}  "
                f"[{done}/{repeats*folds*n_cfg} runs, {time.time()-t0:.0f}s]")
    n_test = len(ds.y) // folds
    return {"acc": acc, "oof": oof, "n_train": len(ds.y) - n_test, "n_test": n_test}


def evaluate(ds, cv: dict, alpha: float = 0.05) -> dict:
    acc, oof = cv["acc"], cv["oof"]
    summary = {name: {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                      "per_repeat": [float(x) for x in a.mean(axis=1)]} for name, a in acc.items()}
    hyps = []
    for h in HYPOTHESES:
        diffs = (acc[h["a"]] - acc[h["b"]]).ravel()
        test = paired_cv_test(diffs, cv["n_train"], cv["n_test"])
        ci = bootstrap_ci(diffs)
        mc = mcnemar_test(ds.y, oof[h["a"]][0], oof[h["b"]][0])
        hyps.append({**h, "test": test, "ci95": ci, "mcnemar_repeat0": mc,
                     "verdict": verdict(test, ci, alpha)})
    best = max(summary, key=lambda n: summary[n]["mean"])
    cm = np.zeros((ds.n_classes, ds.n_classes), dtype=int)
    for yt, yp in zip(ds.y, oof[best][0]):
        cm[yt, yp] += 1
    recall = {g: float(cm[i, i] / cm[i].sum()) for i, g in enumerate(ds.genres)}
    return {"summary": summary, "hypotheses": hyps, "best": best,
            "confusion_repeat0": {"labels": ds.genres, "matrix": cm.tolist()}, "per_genre_recall": recall}


def write_report(out: Path, meta: dict, ev: dict) -> None:
    L = []
    L.append("# Ergebnisbericht: Genre-Klassifikation auf GTZAN-Features (JAX)\n")
    L.append(f"Erzeugt: {meta['timestamp']}  \nBefehl: `{meta['command']}`  \n"
             f"Setup: {meta['repeats']} x {meta['folds']}-fold stratifizierte CV, {meta['steps']} Adam-Schritte, "
             f"Batch {meta['batch_size']}, z-Standardisierung je Fold.  \n"
             f"Umgebung: Python {meta['python']}, JAX {meta['jax']}, {meta['platform']}\n")
    L.append("## Modelle (Accuracy, Mittel +- Std über alle Folds)\n")
    L.append("| Konfiguration | Features | Hidden | Accuracy |\n|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        s = ev["summary"][name]
        L.append(f"| {name} | {len(cfg['features'])} | {cfg['hidden'] or '-'} | {s['mean']:.4f} +- {s['std']:.4f} |")
    L.append("\nZufallsniveau bei 10 balancierten Klassen: 0.1000\n")
    L.append("## Hypothesen\n")
    L.append("Entscheidungsregel (vorregistriert): BELEGT, wenn korrigiertes p < 0.05 (Nadeau-Bengio) und das "
             "95%-Bootstrap-CI der Differenz die 0 ausschliesst, in der behaupteten Richtung.\n")
    for h in ev["hypotheses"]:
        t, (lo, hi), mc = h["test"], h["ci95"], h["mcnemar_repeat0"]
        L.append(f"### {h['id']}: {h['title']}\n")
        L.append(f"Behauptung: accuracy({h['a']}) > accuracy({h['b']})  \n"
                 f"Mittlere Differenz: {t['mean_diff']:+.4f} (Std {t['std_diff']:.4f}, {t['n_pairs']} gepaarte Folds)  \n"
                 f"95%-Bootstrap-CI: [{lo:+.4f}, {hi:+.4f}]  \n"
                 f"t-Test naiv: t={t['t_naive']:.2f}, p={t['p_naive']:.2e}  \n"
                 f"t-Test korrigiert (Nadeau-Bengio): t={t['t_corrected']:.2f}, p={t['p_corrected']:.2e}  \n"
                 f"McNemar (OOF, Wiederholung 0): nur A richtig={mc['only_a_correct']}, "
                 f"nur B richtig={mc['only_b_correct']}, p={mc['p_exact']:.2e}  \n"
                 f"**Urteil: {h['verdict']}**\n")
    L.append(f"## Fehleranalyse bestes Modell ({ev['best']}, Out-of-Fold, Wiederholung 0)\n")
    L.append("| Genre | Recall |\n|---|---|")
    for g, r in sorted(ev["per_genre_recall"].items(), key=lambda kv: kv[1]):
        L.append(f"| {g} | {r:.2f} |")
    labels = ev["confusion_repeat0"]["labels"]
    L.append("\nKonfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt):\n")
    L.append("| | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    for g, row in zip(labels, ev["confusion_repeat0"]["matrix"]):
        L.append(f"| {g} | " + " | ".join(str(v) for v in row) + " |")
    L.append("\n## Einschränkungen\n")
    L.append("- GTZAN enthält bekannte Duplikate, Fehllabels und Artefakte (Sturm 2013); absolute Werte sind "
             "nicht auf andere Datensätze übertragbar, die Paarvergleiche innerhalb des Datensatzes bleiben gültig.\n"
             "- Features sind Clip-Mittelwerte (30 s); zeitliche Struktur ist nicht modelliert.\n"
             "- Hyperparameter sind fix (keine innere CV); ein fairer Vergleich mit Tuning je Modell steht aus.\n")
    L.append("## Reproduktion\n\n```bash\ncd research && pip install -r requirements.txt && python experiment.py\n```\n")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="Pfad zu MGC.csv (Default: ../MGC.csv)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="1 x 3-fold, 60 Schritte (Rauchtest)")
    args = p.parse_args(argv)
    if args.quick:
        args.repeats, args.folds, args.steps = 1, 3, 60

    ds = load_mgc(args.csv) if args.csv else load_mgc()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Daten: {ds.X.shape[0]} Clips, {ds.X.shape[1]} Features, {ds.n_classes} Genres")
    cv = run_cv(ds, repeats=args.repeats, folds=args.folds, steps=args.steps, batch_size=args.batch_size)
    ev = evaluate(ds, cv)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "command": "python " + " ".join(sys.argv),
        "repeats": args.repeats, "folds": args.folds, "steps": args.steps, "batch_size": args.batch_size,
        "python": platform.python_version(), "jax": jax.__version__, "platform": platform.platform(),
        "configs": {n: {**c, "features": len(c["features"])} for n, c in CONFIGS.items()},
    }
    (out / "results.json").write_text(json.dumps({"meta": meta, **ev,
        "fold_accuracies": {n: a.tolist() for n, a in cv["acc"].items()}}, indent=2), encoding="utf-8")
    write_report(out, meta, ev)
    print()
    for h in ev["hypotheses"]:
        print(f"{h['id']}: diff={h['test']['mean_diff']:+.4f} p_corr={h['test']['p_corrected']:.2e} -> {h['verdict']}")
    print(f"\nBericht: {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
