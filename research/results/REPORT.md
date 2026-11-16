# Ergebnisbericht: Genre-Klassifikation auf GTZAN-Features (JAX)

Erzeugt: 2026-11-15 19:17:04 UTC  
Befehl: `python experiment.py --out results`  
Setup: 5 x 10-fold stratifizierte CV, 600 Adam-Schritte, Batch 64, z-Standardisierung je Fold.  
Hyperparameter: innere 3-fold CV je Outer-Fold über 6 Gitterpunkte (lr x l2), 300 Schritte innen.  
Umgebung: Python 3.13.18, JAX 0.12.0.1, Linux-6.18.44-fc-v470-x86_64-with-glibc2.39

## Modelle (Accuracy, Mittel +- Std über alle Folds)

| Konfiguration | Features | Hidden | Accuracy | häufigste Wahl (lr, l2) |
|---|---|---|---|---|
| softmax_all | 26 | - | 0.6180 +- 0.0399 | lr=0.01, l2=0.001 (29x) |
| mlp_all | 26 | [128, 64] | 0.6750 +- 0.0380 | lr=0.01, l2=0.0001 (16x) |
| softmax_mfcc | 20 | - | 0.5516 +- 0.0423 | lr=0.01, l2=0.0001 (25x) |
| mlp_mfcc | 20 | [128, 64] | 0.6109 +- 0.0438 | lr=0.01, l2=0.001 (16x) |

Gewählte Hyperparameter je Konfiguration (Anzahl Outer-Folds):

| Konfiguration | lr=0.001, l2=0.0001 | lr=0.001, l2=0.001 | lr=0.003, l2=0.0001 | lr=0.003, l2=0.001 | lr=0.01, l2=0.0001 | lr=0.01, l2=0.001 |
|---|---|---|---|---|---|---|
| softmax_all | 0 | 0 | 0 | 0 | 21 | 29 |
| mlp_all | 0 | 0 | 6 | 15 | 16 | 13 |
| softmax_mfcc | 0 | 0 | 0 | 0 | 25 | 25 |
| mlp_mfcc | 2 | 2 | 4 | 14 | 12 | 16 |

Zufallsniveau bei 10 annähernd balancierten Klassen: ca. 0.10

### Datenbereinigung

Clips nach Deduplizierung: 985 von 1000.  
Entfernte exakte Kopien (13): disco.00098.au, hiphop.00039.au, hiphop.00076.au, metal.00034.au, metal.00040.au, metal.00041.au, metal.00043.au, metal.00063.au, metal.00065.au, metal.00066.au, pop.00060.au, pop.00071.au, reggae.00081.au  
Entfernt wegen Label-Konflikt bei identischen Features (2): metal.00058.au, rock.00016.au

## Hypothesen

Entscheidungsregel (vorregistriert, symmetrisch): BELEGT, wenn das Holm-adjustierte korrigierte p (Nadeau-Bengio, Familie von 3 Hypothesen) < 0.05 und das korrigierte 95%-CI der Differenz vollständig > 0 liegt. WIDERLEGT bei p_Holm < 0.05 und CI < 0. Sonst NICHT BELEGT. Das unkorrigierte Bootstrap-CI ist nur zur Information angegeben.

### H1: Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell auf den 26 Features

Behauptung: accuracy(mlp_all) > accuracy(softmax_all)  
Mittlere Differenz: +0.0570 (Std 0.0504, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0204, +0.0936]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0433, +0.0708]  
t-Test naiv: t=8.00, p=1.91e-10  
t-Test korrigiert (Nadeau-Bengio): t=3.13, p=2.93e-03  
p Holm-adjustiert: 2.93e-03  
McNemar (OOF, Wiederholung 0): nur A richtig=139, nur B richtig=80, p=8.11e-05  
**Urteil: BELEGT**

### H2: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (MLP)

Behauptung: accuracy(mlp_all) > accuracy(mlp_mfcc)  
Mittlere Differenz: +0.0642 (Std 0.0425, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0333, +0.0950]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0528, +0.0759]  
t-Test naiv: t=10.66, p=2.28e-14  
t-Test korrigiert (Nadeau-Bengio): t=4.18, p=1.22e-04  
p Holm-adjustiert: 2.43e-04  
McNemar (OOF, Wiederholung 0): nur A richtig=142, nur B richtig=55, p=4.69e-10  
**Urteil: BELEGT**

### H3: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (Softmax)

Behauptung: accuracy(softmax_all) > accuracy(softmax_mfcc)  
Mittlere Differenz: +0.0664 (Std 0.0405, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0370, +0.0958]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0554, +0.0772]  
t-Test naiv: t=11.58, p=1.24e-15  
t-Test korrigiert (Nadeau-Bengio): t=4.53, p=3.75e-05  
p Holm-adjustiert: 1.12e-04  
McNemar (OOF, Wiederholung 0): nur A richtig=102, nur B richtig=38, p=6.02e-08  
**Urteil: BELEGT**

## Fehleranalyse bestes Modell (mlp_all, Out-of-Fold, Wiederholung 0)

| Genre | Recall |
|---|---|
| disco | 0.54 |
| reggae | 0.55 |
| rock | 0.55 |
| country | 0.61 |
| hiphop | 0.66 |
| blues | 0.71 |
| jazz | 0.72 |
| pop | 0.77 |
| metal | 0.79 |
| classical | 0.92 |

Konfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt):

| | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 71 | 0 | 8 | 3 | 1 | 6 | 4 | 0 | 0 | 7 |
| classical | 1 | 92 | 1 | 0 | 1 | 3 | 0 | 0 | 1 | 1 |
| country | 8 | 2 | 61 | 4 | 3 | 6 | 0 | 2 | 7 | 7 |
| disco | 4 | 2 | 2 | 53 | 7 | 3 | 4 | 4 | 9 | 11 |
| hiphop | 1 | 2 | 4 | 6 | 65 | 0 | 5 | 2 | 11 | 2 |
| jazz | 2 | 7 | 7 | 2 | 0 | 72 | 0 | 0 | 5 | 5 |
| metal | 8 | 1 | 1 | 1 | 6 | 1 | 73 | 0 | 0 | 1 |
| pop | 0 | 1 | 3 | 9 | 3 | 2 | 0 | 75 | 2 | 3 |
| reggae | 2 | 0 | 8 | 4 | 16 | 6 | 0 | 4 | 54 | 5 |
| rock | 3 | 0 | 6 | 12 | 3 | 7 | 2 | 1 | 11 | 54 |

## Einschränkungen

- GTZAN enthält bekannte Duplikate, Fehllabels und Artefakte (Sturm 2013). Exakte Feature-Duplikate sind oben entfernt; nahezu identische Clips und Fehllabels bleiben. Absolute Werte sind nicht auf andere Datensätze übertragbar, die Paarvergleiche innerhalb des Datensatzes bleiben gültig.
- Features sind Clip-Mittelwerte (30 s); zeitliche Struktur ist nicht modelliert.
- Hyperparameter werden je Outer-Fold per innerer CV aus einem festen Gitter gewählt (bei --no-tune fix). Architektur (Hidden-Grössen) und Schrittzahl bleiben fix.

## Reproduktion

```bash
cd research && pip install -r requirements.txt && python experiment.py
```
