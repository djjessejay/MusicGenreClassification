# H2 / H3: Bringen die 6 spektralen Features einen Gewinn über MFCC-only hinaus?

Status: H2 BELEGT (Differenz +0.064, p_Holm 2.4e-4, CI [+0.033, +0.095]), H3 BELEGT (Differenz +0.066, p_Holm 1.1e-4, CI [+0.037, +0.096]). Lauf 2026-11-15 auf 985 deduplizierten Clips, 5 x 10-fold mit innerer 3-fold CV für lr/l2, Details: results/REPORT.md

## Frage
MFCC-Koeffizienten beschreiben die spektrale Hüllkurve bereits. Tragen rmse, chroma_stft, spec_cent, spec_bw,
rolloff und zcr darüber hinaus Information, oder sind sie redundant?

## Hypothesen (Ablation)
H2: accuracy(mlp_all) > accuracy(mlp_mfcc)
H3: accuracy(softmax_all) > accuracy(softmax_mfcc)
H0 jeweils: kein Unterschied.

Zwei Modellfamilien, damit das Urteil nicht an einer Architektur hängt. Stimmen H2 und H3 überein, ist der
Befund robust. Widersprechen sie sich, ist der Nutzen der Features modellabhängig.

## Begründung
rmse (Lautheit/Dynamik) und zcr (Perkussivität) sind in MFCCs nicht enthalten. chroma_stft trägt tonale
Information, die MFCCs durch die Mel-Glättung weitgehend verwerfen. Erwartung: kleiner, aber messbarer Gewinn.

## Metrik und Entscheidungsregel
Wie H1: gepaarte Accuracy-Differenzen über 5 x 10 Folds, korrigierter t-Test, Bootstrap-CI, McNemar.

## Mögliche Fehlerquellen
- Skalenunterschiede (spec_cent in Hz, rmse in [0,1]): z-Standardisierung je Trainingsfold.
- Korrelierte Features erhöhen Varianz des linearen Modells: L2-Regularisierung aktiv.
