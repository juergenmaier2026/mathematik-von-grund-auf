"""Band 24, Kapitel 10 -- Ziele.

Erzeugt: Nettonutzen dreier Strategien bei zwei Zahlungsbereitschaften (10.3)
und die probabilistische Sensitivitätsanalyse (10.5): für welchen Anteil der
Parameterziehungen jede Strategie gewinnt. Mittelwerte aus dem Buch; die
Unsicherheit (vom Buch nur beschrieben) ist hier als +-10 % angesetzt.
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2410)
strategien = {"nichts": (8.36, 0.0), "Gerinnungshemmung": (9.461, 7_344.0), "neues Mittel": (9.72, 19_000.0)}   # QALY, Kosten

kopf("Nettonutzen = w * QALY - Kosten (10.3)")
zeile("Strategie", "w = 20 000", "w = 50 000", breite=20)
for name, (q, c) in strategien.items():
    zeile(name, round(20_000 * q - c, 0), round(50_000 * q - c, 0), breite=20)
buch("167 200 / 181 876 / 175 400 und 418 000 / 465 706 / 467 000")

kopf("Probabilistische Sensitivitätsanalyse: 10 000 Ziehungen, QALY-Gewinn und Kosten je +-10 % unsicher (10.5)")
N = 10_000
zeile("w", *strategien.keys(), breite=18)
for w in (10_000, 20_000, 30_000, 50_000, 100_000):
    nb = []
    for name, (q, c) in strategien.items():
        gewinn = q - 8.36
        qz = 8.36 + gewinn * rng.normal(1, 0.1, N) if gewinn else np.full(N, 8.36)
        cz = c * rng.normal(1, 0.1, N) if c else np.zeros(N)
        nb.append(w * qz - cz)
    nb = np.array(nb); sieger = nb.argmax(axis=0)
    zeile(w, *[f"{np.mean(sieger == k):.0%}" for k in range(3)], breite=18)
buch("die Kurve der Gewinnwahrscheinlichkeit gegen die Schwelle -- bei 20 000 gewinnt die Gerinnungshemmung, bei 50 000 ist es knapp")
