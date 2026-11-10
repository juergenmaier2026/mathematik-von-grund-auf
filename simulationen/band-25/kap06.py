"""Band 25, Kapitel 6 -- Aussterben oder Ausbruch.

Erzeugt: Aussterbewahrscheinlichkeit bei Poisson-Nachkommen (Satz 6.3) und
bei negativ-binomialer Nachkommenzahl mit Streuparameter k (6.3), jeweils
Fixpunkt der erzeugenden Funktion gegen Simulation.
"""
import _pfad
import math
import numpy as np
from scipy.optimize import brentq
from mvg.prozesse import verzweigung_aussterben
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2506)

kopf("Poissonsche Nachkommenzahl: q = exp(-R0 (1-q)) gegen 20 000 simulierte Linien, 60 Generationen")
zeile("R0", "Fixpunkt", "simuliert", breite=12)
for R0 in (0.9, 1.0, 1.5, 2.0, 3.0):
    q = 1.0 if R0 <= 1 else brentq(lambda s: math.exp(-R0 * (1 - s)) - s, 0, 1 - 1e-9)
    sim = verzweigung_aussterben(rng, lambda r, z: r.poisson(R0 * z), 60, 20_000)      # Summe von z Poisson(R0) = Poisson(R0 z)
    zeile(R0, round(q, 3), round(float(sim), 3), breite=12)
buch("1,000 / 1,000 / 0,417 / 0,203 / 0,060 -- bei R0 = 1 stirbt die Linie sicher aus, aber langsam: nach 60 Generationen erst 97 %")

kopf("R0 = 2 mit Streuparameter k (negative Binomialverteilung): G(s) = (1 + R0 (1-s)/k)^(-k)")
zeile("k", "Fixpunkt", "simuliert", breite=12)
for k in (0.1, 0.5, 1.0, 10.0):
    q = brentq(lambda s: (1 + 2 * (1 - s) / k) ** (-k) - s, 0, 1 - 1e-9)
    nb = lambda r, z, k=k: r.negative_binomial(k * z, k / (k + 2))          # Summe von z NB(k, p) = NB(k z, p)
    sim = verzweigung_aussterben(rng, nb, 60, 20_000)
    zeile(k, round(q, 3), round(float(sim), 3), breite=12)
buch("0,890 / 0,640 / 0,500 / 0,245 -- je ungleicher die Verteilung, desto öfter stirbt die Linie aus")
