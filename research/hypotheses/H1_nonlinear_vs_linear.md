# H1: Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell

Status: BELEGT (Lauf 2026-11-15 auf 985 deduplizierten Clips, 5 x 10-fold mit innerer 3-fold CV für lr/l2, Differenz +0.057, p_Holm 2.9e-3, korrigiertes CI [+0.020, +0.094]). Details: results/REPORT.md

## Frage
Reichen die 26 Clip-Mittelwert-Features (6 spektrale + 20 MFCC) für lineare Trennbarkeit der 10 GTZAN-Genres,
oder braucht es nichtlineare Entscheidungsgrenzen?

## Hypothese
H1: accuracy(mlp_all) > accuracy(softmax_all)
H0: kein Unterschied oder Softmax besser.

## Begründung
Genres wie metal/rock oder disco/pop überlappen in Mittelwert-Features. Ein zweischichtiges MLP (128-64, ReLU)
kann Wechselwirkungen zwischen MFCC-Koeffizienten und spektralem Schwerpunkt abbilden, Softmax-Regression nicht.

## Metrik und Entscheidungsregel
- Accuracy, 5 x 10-fold stratifizierte CV, identische Folds für beide Modelle (gepaart).
- BELEGT, wenn p_korrigiert (Nadeau-Bengio) < 0.05 und 95%-Bootstrap-CI der Differenz > 0.
- Zusätzlich McNemar auf Out-of-Fold-Vorhersagen als fold-unabhängige Kontrolle.

## Konfiguration
softmax_all: Softmax-Regression. mlp_all: hidden [128, 64]. lr und l2 je Outer-Fold per innerer 3-fold CV aus GRID (lr 1e-3/3e-3/1e-2 x l2 1e-4/1e-3). 600 Adam-Schritte, Batch 64.

## Mögliche Fehlerquellen
- Softmax unterfitted bei zu wenigen Schritten: Sanity-Check über Trainingsverlust in tests/.
- MLP overfittet bei 900 Trainingsclips: L2 und frühes Stoppen via feste Schrittzahl.
