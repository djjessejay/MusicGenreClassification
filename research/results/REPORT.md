# Ergebnisbericht: Genre-Klassifikation auf GTZAN-Features (JAX)

Erzeugt: 2026-11-15 18:15:51 UTC  
Befehl: `python experiment.py --out results`  
Setup: 5 x 10-fold stratifizierte CV, 600 Adam-Schritte, Batch 64, z-Standardisierung je Fold.  
Umgebung: Python 3.13.18, JAX 0.12.0.1, Linux-6.18.44-fc-v470-x86_64-with-glibc2.39

## Modelle (Accuracy, Mittel +- Std über alle Folds)

| Konfiguration | Features | Hidden | Accuracy |
|---|---|---|---|
| softmax_all | 26 | - | 0.6070 +- 0.0405 |
| mlp_all | 26 | [128, 64] | 0.6644 +- 0.0485 |
| softmax_mfcc | 20 | - | 0.5504 +- 0.0451 |
| mlp_mfcc | 20 | [128, 64] | 0.6236 +- 0.0373 |

Zufallsniveau bei 10 balancierten Klassen: 0.1000

## Hypothesen

Entscheidungsregel (vorregistriert): BELEGT, wenn korrigiertes p < 0.05 (Nadeau-Bengio) und das 95%-Bootstrap-CI der Differenz die 0 ausschliesst, in der behaupteten Richtung.

### H1: Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell auf den 26 Features

Behauptung: accuracy(mlp_all) > accuracy(softmax_all)  
Mittlere Differenz: +0.0574 (Std 0.0456, 50 gepaarte Folds)  
95%-Bootstrap-CI: [+0.0450, +0.0700]  
t-Test naiv: t=8.91, p=8.13e-12  
t-Test korrigiert (Nadeau-Bengio): t=3.48, p=1.07e-03  
McNemar (OOF, Wiederholung 0): nur A richtig=140, nur B richtig=85, p=2.99e-04  
**Urteil: BELEGT**

### H2: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (MLP)

Behauptung: accuracy(mlp_all) > accuracy(mlp_mfcc)  
Mittlere Differenz: +0.0408 (Std 0.0408, 50 gepaarte Folds)  
95%-Bootstrap-CI: [+0.0300, +0.0520]  
t-Test naiv: t=7.07, p=5.15e-09  
t-Test korrigiert (Nadeau-Bengio): t=2.76, p=8.07e-03  
McNemar (OOF, Wiederholung 0): nur A richtig=115, nur B richtig=67, p=4.62e-04  
**Urteil: BELEGT**

### H3: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (Softmax)

Behauptung: accuracy(softmax_all) > accuracy(softmax_mfcc)  
Mittlere Differenz: +0.0566 (Std 0.0343, 50 gepaarte Folds)  
95%-Bootstrap-CI: [+0.0470, +0.0662]  
t-Test naiv: t=11.68, p=9.05e-16  
t-Test korrigiert (Nadeau-Bengio): t=4.56, p=3.41e-05  
McNemar (OOF, Wiederholung 0): nur A richtig=100, nur B richtig=35, p=1.94e-08  
**Urteil: BELEGT**

## Fehleranalyse bestes Modell (mlp_all, Out-of-Fold, Wiederholung 0)

| Genre | Recall |
|---|---|
| rock | 0.49 |
| disco | 0.52 |
| reggae | 0.54 |
| country | 0.59 |
| hiphop | 0.67 |
| blues | 0.71 |
| jazz | 0.71 |
| pop | 0.79 |
| metal | 0.80 |
| classical | 0.89 |

Konfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt):

| | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 71 | 0 | 9 | 1 | 0 | 5 | 8 | 0 | 3 | 3 |
| classical | 1 | 89 | 1 | 1 | 0 | 6 | 0 | 0 | 1 | 1 |
| country | 8 | 0 | 59 | 3 | 1 | 7 | 0 | 5 | 10 | 7 |
| disco | 5 | 1 | 1 | 52 | 11 | 1 | 6 | 6 | 7 | 10 |
| hiphop | 0 | 0 | 6 | 7 | 67 | 0 | 3 | 3 | 13 | 1 |
| jazz | 2 | 5 | 11 | 2 | 0 | 71 | 1 | 0 | 5 | 3 |
| metal | 7 | 0 | 1 | 2 | 5 | 1 | 80 | 0 | 0 | 4 |
| pop | 0 | 1 | 4 | 4 | 5 | 0 | 0 | 79 | 3 | 4 |
| reggae | 2 | 1 | 8 | 4 | 13 | 4 | 1 | 6 | 54 | 7 |
| rock | 8 | 0 | 7 | 14 | 2 | 7 | 5 | 1 | 7 | 49 |

## Einschränkungen

- GTZAN enthält bekannte Duplikate, Fehllabels und Artefakte (Sturm 2013); absolute Werte sind nicht auf andere Datensätze übertragbar, die Paarvergleiche innerhalb des Datensatzes bleiben gültig.
- Features sind Clip-Mittelwerte (30 s); zeitliche Struktur ist nicht modelliert.
- Hyperparameter sind fix (keine innere CV); ein fairer Vergleich mit Tuning je Modell steht aus.

## Reproduktion

```bash
cd research && pip install -r requirements.txt && python experiment.py
```
