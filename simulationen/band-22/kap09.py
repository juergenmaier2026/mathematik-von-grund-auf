"""Band 22, Kapitel 9 -- Bäume.

Erzeugt: die Instabilität des Baums (9.3): 200 Bäume auf Bootstrap-Stichproben
von 500 Patienten mit sieben Merkmalen, zwei davon reine Zufallszahlen.
Daten (vom Buch nur beschrieben): Sepsis-Wahrscheinlichkeit aus Laktat, Atemfrequenz und Temperatur.
"""
import _pfad
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2209)
n = 500
namen = ["Temperatur", "Herzfrequenz", "Laktat", "Atemfrequenz", "Alter", "Zufall 1", "Zufall 2"]
X = np.column_stack([rng.normal(37.5, 0.8, n), rng.normal(90, 15, n), rng.lognormal(0.6, 0.5, n),
                     rng.normal(20, 5, n), rng.normal(60, 15, n), rng.standard_normal(n), rng.standard_normal(n)])
logit = -1.2 + 2.0 * (X[:, 2] - 2.3) + 0.15 * (X[:, 3] - 20) + 0.5 * (X[:, 0] - 37.5)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
print(f"Sepsis bei {y.mean():.0%} der Patienten")

kopf("200 Bäume (Tiefe 3) auf Bootstrap-Stichproben: welches Merkmal fragt die Wurzel, welches der rechte Ast? (9.3)")
wurzel, rechts, schwellen = {}, {}, []
for _ in range(200):
    idx = rng.integers(0, n, n)
    baum = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[idx], y[idx]).tree_
    wurzel[namen[baum.feature[0]]] = wurzel.get(namen[baum.feature[0]], 0) + 1
    if namen[baum.feature[0]] == "Laktat": schwellen.append(baum.threshold[0])
    r = baum.children_right[0]
    if baum.feature[r] >= 0: rechts[namen[baum.feature[r]]] = rechts.get(namen[baum.feature[r]], 0) + 1
print("Wurzel:       ", dict(sorted(wurzel.items(), key=lambda kv: -kv[1])))
print(f"Laktatschwelle: 90 % der Fälle zwischen {np.percentile(schwellen, 5):.2f} und {np.percentile(schwellen, 95):.2f}")
print("rechter Ast:  ", dict(sorted(rechts.items(), key=lambda kv: -kv[1])))
buch("Wurzel 199-mal Laktat; Schwelle 2,04-2,67; rechter Ast 142-mal Atemfrequenz, 43-mal eine Zufallszahl")
