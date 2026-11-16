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
from mgc_jax.stats import bootstrap_ci, holm_correction, mcnemar_test, paired_cv_test, verdict

# --------------------------------------------------------------------------------------
# Konfigurationen (Name -> Featureset + Architektur + Default-Optimierer). sizes[0] wird gesetzt.
# lr und l2 sind nur der Fallback für --no-tune; mit innerer CV werden sie je Outer-Fold aus GRID gewählt.
# --------------------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "softmax_all":  {"features": ALL_FEATURES,  "hidden": [],        "lr": 5e-3, "l2": 1e-4},
    "mlp_all":      {"features": ALL_FEATURES,  "hidden": [128, 64], "lr": 1e-3, "l2": 1e-4},
    "softmax_mfcc": {"features": MFCC_FEATURES, "hidden": [],        "lr": 5e-3, "l2": 1e-4},
    "mlp_mfcc":     {"features": MFCC_FEATURES, "hidden": [128, 64], "lr": 1e-3, "l2": 1e-4},
}

# Hyperparameter-Gitter der inneren CV, identisch für alle Konfigurationen (vorregistriert).
GRID: list[tuple[float, float]] = [(lr, l2) for lr in (1e-3, 3e-3, 1e-2) for l2 in (1e-4, 1e-3)]

# Vorregistrierte Hypothesen. Behauptung ist immer: accuracy(a) > accuracy(b).
HYPOTHESES: list[dict] = [
    {"id": "H1", "a": "mlp_all", "b": "softmax_all",
     "title": "Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell auf den 26 Features"},
    {"id": "H2", "a": "mlp_all", "b": "mlp_mfcc",
     "title": "Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (MLP)"},
    {"id": "H3", "a": "softmax_all", "b": "softmax_mfcc",
     "title": "Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (Softmax)"},
]


def tune_hyperparams(X: np.ndarray, y: np.ndarray, hidden: list[int], n_classes: int, *,
                     grid: list[tuple[float, float]], inner_folds: int, steps: int, batch_size: int,
                     seed: int) -> tuple[float, float]:
    """Innere stratifizierte CV ausschliesslich auf dem äusseren Trainingsfold. Wählt (lr, l2) mit der
    höchsten mittleren inneren Accuracy. Der äussere Testfold wird hier nie berührt."""
    scores = []
    for lr, l2 in grid:
        inner = []
        for j, (itr, ite) in enumerate(stratified_folds(y, inner_folds, seed=seed)):
            Xi_tr, Xi_te = standardize(X[itr], X[ite])
            params = train_classifier(Xi_tr, y[itr], [X.shape[1], *hidden, n_classes], steps=steps,
                                      batch_size=batch_size, lr=lr, l2=l2, seed=seed * 7 + j)
            inner.append(float(np.mean(predict(params, Xi_te) == y[ite])))
        scores.append(float(np.mean(inner)))
    return grid[int(np.argmax(scores))]


def run_cv(ds, *, repeats: int, folds: int, steps: int, batch_size: int, tune: bool = True,
           inner_folds: int = 3, log=print) -> dict:
    n_cfg = len(CONFIGS)
    acc = {name: np.zeros((repeats, folds)) for name in CONFIGS}
    oof = {name: np.zeros((repeats, len(ds.y)), dtype=np.int64) for name in CONFIGS}
    chosen = {name: [] for name in CONFIGS}   # (lr, l2) je (repeat, fold)
    t0 = time.time()
    for r in range(repeats):
        for k, (tr, te) in enumerate(stratified_folds(ds.y, folds, seed=1000 + r)):
            for name, cfg in CONFIGS.items():
                sub = ds.select(cfg["features"])
                seed = r * folds + k
                if tune:
                    lr, l2 = tune_hyperparams(sub.X[tr], ds.y[tr], cfg["hidden"], ds.n_classes, grid=GRID,
                                              inner_folds=inner_folds, steps=max(steps // 2, 20),
                                              batch_size=batch_size, seed=seed)
                else:
                    lr, l2 = cfg["lr"], cfg["l2"]
                chosen[name].append((lr, l2))
                Xtr, Xte = standardize(sub.X[tr], sub.X[te])
                sizes = [Xtr.shape[1], *cfg["hidden"], ds.n_classes]
                params = train_classifier(Xtr, ds.y[tr], sizes, steps=steps, batch_size=batch_size,
                                          lr=lr, l2=l2, seed=seed)
                pred = predict(params, Xte)
                acc[name][r, k] = float(np.mean(pred == ds.y[te]))
                oof[name][r, te] = pred
            done = (r * folds + k + 1) * n_cfg
            log(f"  repeat {r+1}/{repeats} fold {k+1}/{folds}  "
                f"{'  '.join(f'{n}={acc[n][r,k]:.3f}' for n in CONFIGS)}  "
                f"[{done}/{repeats*folds*n_cfg} runs, {time.time()-t0:.0f}s]", flush=True)
    n_test = int(round(len(ds.y) / folds))
    return {"acc": acc, "oof": oof, "chosen": chosen, "n_train": len(ds.y) - n_test, "n_test": n_test}


def evaluate(ds, cv: dict, alpha: float = 0.05) -> dict:
    acc, oof = cv["acc"], cv["oof"]
    summary = {name: {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                      "per_repeat": [float(x) for x in a.mean(axis=1)]} for name, a in acc.items()}
    hyps = []
    for h in HYPOTHESES:
        diffs = (acc[h["a"]] - acc[h["b"]]).ravel()
        test = paired_cv_test(diffs, cv["n_train"], cv["n_test"], alpha=alpha)
        mc = mcnemar_test(ds.y, oof[h["a"]][0], oof[h["b"]][0])
        hyps.append({**h, "test": test, "ci95_bootstrap_uncorrected": bootstrap_ci(diffs), "mcnemar_repeat0": mc})
    # Mehrfachtests: Holm über die Familie aller vorregistrierten Hypothesen dieses Laufs.
    for h, p_holm in zip(hyps, holm_correction([h["test"]["p_corrected"] for h in hyps])):
        h["test"]["p_holm"] = p_holm
        h["verdict"] = verdict(h["test"], p_key="p_holm")
    hp_counts = {}
    for name, cs in cv["chosen"].items():
        counts: dict[str, int] = {}
        for lr, l2 in cs:
            key = f"lr={lr:g}, l2={l2:g}"
            counts[key] = counts.get(key, 0) + 1
        hp_counts[name] = counts
    best = max(summary, key=lambda n: summary[n]["mean"])
    cm = np.zeros((ds.n_classes, ds.n_classes), dtype=int)
    for yt, yp in zip(ds.y, oof[best][0]):
        cm[yt, yp] += 1
    recall = {g: float(cm[i, i] / cm[i].sum()) for i, g in enumerate(ds.genres)}
    return {"summary": summary, "hypotheses": hyps, "best": best, "hyperparams_chosen": hp_counts,
            "confusion_repeat0": {"labels": ds.genres, "matrix": cm.tolist()}, "per_genre_recall": recall}


def write_report(out: Path, meta: dict, ev: dict) -> None:
    L = []
    L.append("# Ergebnisbericht: Genre-Klassifikation auf GTZAN-Features (JAX)\n")
    L.append(f"Erzeugt: {meta['timestamp']}  \nBefehl: `{meta['command']}`  \n"
             f"Setup: {meta['repeats']} x {meta['folds']}-fold stratifizierte CV, {meta['steps']} Adam-Schritte, "
             f"Batch {meta['batch_size']}, z-Standardisierung je Fold.  \n"
             f"Hyperparameter: {'innere ' + str(meta['inner_folds']) + '-fold CV je Outer-Fold über ' + str(len(GRID)) + ' Gitterpunkte (lr x l2), ' + str(max(meta['steps'] // 2, 20)) + ' Schritte innen' if meta['tune'] else 'fix aus CONFIGS (kein Tuning)'}.  \n"
             f"Umgebung: Python {meta['python']}, JAX {meta['jax']}, {meta['platform']}\n")
    L.append("## Modelle (Accuracy, Mittel +- Std über alle Folds)\n")
    L.append("| Konfiguration | Features | Hidden | Accuracy | häufigste Wahl (lr, l2) |\n|---|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        s = ev["summary"][name]
        top = max(ev["hyperparams_chosen"][name].items(), key=lambda kv: kv[1])
        L.append(f"| {name} | {len(cfg['features'])} | {cfg['hidden'] or '-'} | {s['mean']:.4f} +- {s['std']:.4f} "
                 f"| {top[0]} ({top[1]}x) |")
    L.append("\nGewählte Hyperparameter je Konfiguration (Anzahl Outer-Folds):\n")
    L.append("| Konfiguration | " + " | ".join(f"lr={lr:g}, l2={l2:g}" for lr, l2 in GRID) + " |\n|---|" + "---|" * len(GRID))
    for name in CONFIGS:
        L.append(f"| {name} | " + " | ".join(str(ev["hyperparams_chosen"][name].get(f"lr={lr:g}, l2={l2:g}", 0)) for lr, l2 in GRID) + " |")
    L.append("\nZufallsniveau bei 10 annähernd balancierten Klassen: ca. 0.10\n")
    L.append("### Datenbereinigung\n")
    L.append(f"Clips nach Deduplizierung: {meta['n_clips']} von {meta['n_raw']}.  \n"
             f"Entfernte exakte Kopien ({len(meta['dropped_duplicates'])}): {', '.join(meta['dropped_duplicates'])}  \n"
             f"Entfernt wegen Label-Konflikt bei identischen Features ({len(meta['dropped_conflicts'])}): "
             f"{', '.join(meta['dropped_conflicts'])}\n")
    L.append("## Hypothesen\n")
    L.append(f"Entscheidungsregel (vorregistriert, symmetrisch): BELEGT, wenn das Holm-adjustierte korrigierte p "
             f"(Nadeau-Bengio, Familie von {len(ev['hypotheses'])} Hypothesen) < 0.05 und das korrigierte 95%-CI der "
             "Differenz vollständig > 0 liegt. WIDERLEGT bei p_Holm < 0.05 und CI < 0. Sonst NICHT BELEGT. "
             "Das unkorrigierte Bootstrap-CI ist nur zur Information angegeben.\n")
    for h in ev["hypotheses"]:
        t, mc = h["test"], h["mcnemar_repeat0"]
        (lo, hi), (blo, bhi) = t["ci95_corrected"], h["ci95_bootstrap_uncorrected"]
        L.append(f"### {h['id']}: {h['title']}\n")
        L.append(f"Behauptung: accuracy({h['a']}) > accuracy({h['b']})  \n"
                 f"Mittlere Differenz: {t['mean_diff']:+.4f} (Std {t['std_diff']:.4f}, {t['n_pairs']} gepaarte Folds)  \n"
                 f"95%-CI korrigiert (Nadeau-Bengio): [{lo:+.4f}, {hi:+.4f}]  \n"
                 f"95%-CI Bootstrap unkorrigiert (nur Info): [{blo:+.4f}, {bhi:+.4f}]  \n"
                 f"t-Test naiv: t={t['t_naive']:.2f}, p={t['p_naive']:.2e}  \n"
                 f"t-Test korrigiert (Nadeau-Bengio): t={t['t_corrected']:.2f}, p={t['p_corrected']:.2e}  \n"
                 f"p Holm-adjustiert: {t['p_holm']:.2e}  \n"
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
    L.append("- GTZAN enthält bekannte Duplikate, Fehllabels und Artefakte (Sturm 2013). Exakte Feature-Duplikate "
             "sind oben entfernt; nahezu identische Clips und Fehllabels bleiben. Absolute Werte sind nicht auf andere "
             "Datensätze übertragbar, die Paarvergleiche innerhalb des Datensatzes bleiben gültig.\n"
             "- Features sind Clip-Mittelwerte (30 s); zeitliche Struktur ist nicht modelliert.\n"
             "- Hyperparameter werden je Outer-Fold per innerer CV aus einem festen Gitter gewählt (bei --no-tune fix). "
             "Architektur (Hidden-Grössen) und Schrittzahl bleiben fix.\n")
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
    p.add_argument("--no-tune", dest="tune", action="store_false",
                   help="feste lr/l2 aus CONFIGS statt innerer CV über GRID")
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="1 x 3-fold, 60 Schritte, 2 innere Folds (Rauchtest)")
    args = p.parse_args(argv)
    if args.quick:
        args.repeats, args.folds, args.steps, args.inner_folds = 1, 3, 60, 2

    ds = load_mgc(args.csv) if args.csv else load_mgc()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Daten: {ds.X.shape[0]} Clips, {ds.X.shape[1]} Features, {ds.n_classes} Genres "
          f"(entfernt: {len(ds.dropped_duplicates)} Duplikate, {len(ds.dropped_conflicts)} Label-Konflikte)")
    cv = run_cv(ds, repeats=args.repeats, folds=args.folds, steps=args.steps, batch_size=args.batch_size,
                tune=args.tune, inner_folds=args.inner_folds)
    ev = evaluate(ds, cv)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "command": "python " + " ".join(sys.argv),
        "repeats": args.repeats, "folds": args.folds, "steps": args.steps, "batch_size": args.batch_size,
        "tune": args.tune, "inner_folds": args.inner_folds, "grid": GRID,
        "python": platform.python_version(), "jax": jax.__version__, "platform": platform.platform(),
        "n_clips": int(len(ds.y)),
        "n_raw": int(len(ds.y) + len(ds.dropped_duplicates) + len(ds.dropped_conflicts)),
        "dropped_duplicates": list(ds.dropped_duplicates), "dropped_conflicts": list(ds.dropped_conflicts),
        "configs": {n: {**c, "features": len(c["features"])} for n, c in CONFIGS.items()},
    }
    (out / "results.json").write_text(json.dumps({"meta": meta, **ev,
        "fold_accuracies": {n: a.tolist() for n, a in cv["acc"].items()}}, indent=2), encoding="utf-8")
    write_report(out, meta, ev)
    print()
    for h in ev["hypotheses"]:
        print(f"{h['id']}: diff={h['test']['mean_diff']:+.4f} p_corr={h['test']['p_corrected']:.2e} "
              f"p_holm={h['test']['p_holm']:.2e} -> {h['verdict']}")
    print(f"\nBericht: {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
