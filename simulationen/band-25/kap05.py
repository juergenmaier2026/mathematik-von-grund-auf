"""Band 25, Kapitel 5 -- Warteschlangen.

Erzeugt: Wartezeit gegen Auslastung im M/M/1 (5.3), simuliert gegen die Formel
rho/(1-rho) * Bedienzeit -- mit der Intensivstation als Ein-Bett-Wartesystem.
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

kopf("M/M/1: mittlere Wartezeit gegen Auslastung, Bedienzeit 1 (5.3)")
zeile("rho", "Formel", "simuliert", breite=12)
for rho in (0.5, 0.8, 0.9, 0.95):
    # exponentielle Bedienzeit mit Mittel 1: lognormal mit sigma -> 0 ist das nicht; deshalb eigene kleine Schleife
    rng = np.random.default_rng(2505)
    t = ende = 0.0; warte = []
    for _ in range(200_000):
        t += rng.exponential(1 / rho)
        start = max(t, ende); warte.append(start - t); ende = start + rng.exponential(1.0)
    zeile(rho, round(rho / (1 - rho), 2), round(float(np.mean(warte[20_000:])), 2), breite=12)
buch("die letzten fünf Prozent Auslastung kosten mehr Wartezeit als die ersten neunzig")
