# Ergebnisbericht: Genre-Klassifikation auf GTZAN-Features (JAX)

Erzeugt: 2026-11-15 18:26:41 UTC  
Befehl: `python experiment.py --out results`  
Setup: 5 x 10-fold stratifizierte CV, 600 Adam-Schritte, Batch 64, z-Standardisierung je Fold.  
Umgebung: Python 3.13.18, JAX 0.12.0.1, Linux-6.18.44-fc-v470-x86_64-with-glibc2.39

## Modelle (Accuracy, Mittel +- Std über alle Folds)

| Konfiguration | Features | Hidden | Accuracy |
|---|---|---|---|
| softmax_all | 26 | - | 0.6062 +- 0.0446 |
| mlp_all | 26 | [128, 64] | 0.6620 +- 0.0411 |
| softmax_mfcc | 20 | - | 0.5485 +- 0.0447 |
| mlp_mfcc | 20 | [128, 64] | 0.6093 +- 0.0432 |

Zufallsniveau bei 10 annähernd balancierten Klassen: ca. 0.10

### Datenbereinigung

Clips nach Deduplizierung: 985 von 1000.  
Entfernte exakte Kopien (13): disco.00098.au, hiphop.00039.au, hiphop.00076.au, metal.00034.au, metal.00040.au, metal.00041.au, metal.00043.au, metal.00063.au, metal.00065.au, metal.00066.au, pop.00060.au, pop.00071.au, reggae.00081.au  
Entfernt wegen Label-Konflikt bei identischen Features (2): metal.00058.au, rock.00016.au

## Hypothesen

Entscheidungsregel (vorregistriert, symmetrisch): BELEGT, wenn korrigiertes p < 0.05 (Nadeau-Bengio) und das korrigierte 95%-CI der Differenz vollständig > 0 liegt. WIDERLEGT bei p < 0.05 und CI < 0. Sonst NICHT BELEGT. Das unkorrigierte Bootstrap-CI ist nur zur Information angegeben.

### H1: Nichtlineare Entscheidungsgrenzen schlagen das lineare Modell auf den 26 Features

Behauptung: accuracy(mlp_all) > accuracy(softmax_all)  
Mittlere Differenz: +0.0558 (Std 0.0362, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0295, +0.0820]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0459, +0.0660]  
t-Test naiv: t=10.90, p=1.08e-14  
t-Test korrigiert (Nadeau-Bengio): t=4.27, p=9.06e-05  
McNemar (OOF, Wiederholung 0): nur A richtig=131, nur B richtig=73, p=5.92e-05  
**Urteil: BELEGT**

### H2: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (MLP)

Behauptung: accuracy(mlp_all) > accuracy(mlp_mfcc)  
Mittlere Differenz: +0.0527 (Std 0.0410, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0230, +0.0824]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0417, +0.0641]  
t-Test naiv: t=9.09, p=4.27e-12  
t-Test korrigiert (Nadeau-Bengio): t=3.56, p=8.35e-04  
McNemar (OOF, Wiederholung 0): nur A richtig=129, nur B richtig=57, p=1.36e-07  
**Urteil: BELEGT**

### H3: Die 6 spektralen Zusatzfeatures bringen einen Gewinn über MFCC-only hinaus (Softmax)

Behauptung: accuracy(softmax_all) > accuracy(softmax_mfcc)  
Mittlere Differenz: +0.0577 (Std 0.0383, 50 gepaarte Folds)  
95%-CI korrigiert (Nadeau-Bengio): [+0.0299, +0.0856]  
95%-CI Bootstrap unkorrigiert (nur Info): [+0.0474, +0.0682]  
t-Test naiv: t=10.65, p=2.37e-14  
t-Test korrigiert (Nadeau-Bengio): t=4.17, p=1.24e-04  
McNemar (OOF, Wiederholung 0): nur A richtig=108, nur B richtig=43, p=1.23e-07  
**Urteil: BELEGT**

## Fehleranalyse bestes Modell (mlp_all, Out-of-Fold, Wiederholung 0)

| Genre | Recall |
|---|---|
| rock | 0.47 |
| reggae | 0.53 |
| disco | 0.59 |
| country | 0.59 |
| hiphop | 0.63 |
| blues | 0.72 |
| jazz | 0.72 |
| metal | 0.77 |
| pop | 0.78 |
| classical | 0.92 |

Konfusionsmatrix (Zeile = wahr, Spalte = vorhergesagt):

| | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 72 | 0 | 6 | 1 | 0 | 5 | 8 | 0 | 0 | 8 |
| classical | 1 | 92 | 1 | 0 | 1 | 3 | 0 | 0 | 1 | 1 |
| country | 9 | 0 | 59 | 3 | 2 | 8 | 0 | 7 | 6 | 6 |
| disco | 3 | 1 | 0 | 58 | 8 | 1 | 5 | 4 | 6 | 13 |
| hiphop | 0 | 0 | 4 | 8 | 62 | 0 | 6 | 4 | 11 | 3 |
| jazz | 4 | 6 | 9 | 3 | 0 | 72 | 0 | 0 | 3 | 3 |
| metal | 7 | 0 | 1 | 3 | 7 | 1 | 71 | 0 | 0 | 2 |
| pop | 0 | 1 | 3 | 6 | 3 | 2 | 0 | 76 | 4 | 3 |
| reggae | 3 | 0 | 10 | 5 | 15 | 1 | 0 | 5 | 52 | 8 |
| rock | 6 | 0 | 7 | 15 | 5 | 6 | 3 | 1 | 9 | 47 |

## Einschränkungen

- GTZAN enthält bekannte Duplikate, Fehllabels und Artefakte (Sturm 2013). Exakte Feature-Duplikate sind oben entfernt; nahezu identische Clips und Fehllabels bleiben. Absolute Werte sind nicht auf andere Datensätze übertragbar, die Paarvergleiche innerhalb des Datensatzes bleiben gültig.
- Features sind Clip-Mittelwerte (30 s); zeitliche Struktur ist nicht modelliert.
- Hyperparameter sind fix (keine innere CV); ein fairer Vergleich mit Tuning je Modell steht aus.

## Reproduktion

```bash
cd research && pip install -r requirements.txt && python experiment.py
```
