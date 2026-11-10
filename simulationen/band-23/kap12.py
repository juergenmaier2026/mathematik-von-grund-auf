"""Band 23, Kapitel 12 -- Zeit.

Erzeugt: zeitveränderliche Störgrößen -- warum eine Regression den Effekt einer
Behandlung über zwei Zeitpunkte nicht findet und die g-Formel es tut (12.2).
Modell (vom Buch nur beschrieben): Laktat L0 -> Behandlung A0 -> Laktat L1 -> Behandlung A1 -> Ausgang Y;
jede Behandlung senkt Y um 1, L1 hängt von A0 und von einer unbeobachteten Schwere U ab.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import kq
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2312)
n = 500_000
U = rng.standard_normal(n)
L0 = U + rng.standard_normal(n)
A0 = (rng.random(n) < 1 / (1 + np.exp(-L0))).astype(float)
L1 = U - 0.8 * A0 + rng.standard_normal(n)
A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
Y = 2 * U - 1.0 * A0 - 1.0 * A1 + rng.standard_normal(n)        # wahre Wirkung "immer behandeln" gegen "nie": -2

kopf("Wirkung von 'immer behandeln' gegen 'nie behandeln', wahr -2 (12.2)")
b_ohne = kq(Y, A0, A1)[0]
b_mit = kq(Y, A0, A1, L0, L1)[0]
zeile("Regression ohne L", round(b_ohne[1] + b_ohne[2], 2), breite=30)
zeile("Regression mit L0 und L1", round(b_mit[1] + b_mit[2], 2), breite=30)
# g-Formel: E[Y | A0=a0, A1=a1, L0, L1] gemittelt über die Verteilung von L1 gegeben (L0, a0) und L0
def g_formel(a0, a1):
    bY = kq(Y, A0, A1, L0, L1)[0]
    bL = kq(L1, A0, L0)[0]                          # L1 aus A0 und L0 (linear)
    L1_sim = bL[0] + bL[1] * a0 + bL[2] * L0        # Erwartung von L1 unter der Strategie
    return np.mean(bY[0] + bY[1] * a0 + bY[2] * a1 + bY[3] * L0 + bY[4] * L1_sim)
zeile("g-Formel", round(g_formel(1, 1) - g_formel(0, 0), 2), breite=30)
buch("die Regression kann L1 nicht zugleich als Störgröße (für A1) und als Mediator (für A0) behandeln; die g-Formel schon")
