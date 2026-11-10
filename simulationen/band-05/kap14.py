"""Band 5, Kapitel 14 -- Monte-Carlo-Verfahren.

Erzeugt: pi durch Würfeln (Aufgabe 14.2) -- wie viele Punkte für drei Stellen --,
und die Sobol-Folge als Vergleich (14.3).
"""
import _pfad
import math
import numpy as np
from scipy.stats import qmc
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(514)

kopf("Aufgabe 14.2 -- pi als 4 * Trefferanteil im Viertelkreis")
p = math.pi / 4; sigma = math.sqrt(p * (1 - p))
n_noetig = (1.96 * 4 * sigma / 0.0005) ** 2
print(f"Standardfehler von 4 p-Dach: 4 sqrt(p(1-p)/n) = {4 * sigma:.3f}/sqrt(n); für +-0,0005 (drei Stellen, 95 %): n >= {n_noetig:,.0f}")
zeile("n", "Schätzung", "Fehler", breite=14)
for n in (10 ** 3, 10 ** 5, 10 ** 7):
    u = rng.random((n, 2)); sch = 4 * np.mean((u ** 2).sum(axis=1) <= 1)
    zeile(n, round(float(sch), 5), f"{abs(sch - math.pi):.1e}", breite=14)
buch("etwa 41 Millionen Punkte für drei Nachkommastellen; die Trapezregel schafft 15 Stellen mit 20 Auswertungen")

kopf("Quasi-Monte-Carlo: dieselbe Schätzung mit einer Sobol-Folge (14.3)")
zeile("n", "Monte Carlo", "Sobol", breite=14)
for m in (10, 14, 17):
    n = 2 ** m
    mc = 4 * np.mean((rng.random((n, 2)) ** 2).sum(axis=1) <= 1)
    s = 4 * np.mean((qmc.Sobol(2, scramble=True, seed=1).random(n) ** 2).sum(axis=1) <= 1)
    zeile(n, f"{abs(mc - math.pi):.1e}", f"{abs(s - math.pi):.1e}", breite=14)
buch("dieselbe Genauigkeit mit etwa 10^5 statt 4e7 Punkten -- bei einer Unstetigkeit (Kreisrand) hilft Sobol weniger als bei glatten Integranden")
