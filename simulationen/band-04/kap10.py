"""Band 4, Kapitel 4, 10 und 12 -- Inversion, Monte Carlo, Markow-Ketten-Monte-Carlo.

Band 4 beschreibt die Verfahren, ohne Tabellen; dieses Skript zeigt sie an je einem
Beispiel und verweist auf Band 27, wo sie ausgeführt sind (Kapitel 2, 1 und 7).
"""
import _pfad
import math
import numpy as np
from mvg.zufall import box_muller
from mvg.mcmc import metropolis_zufallsweg
from mvg.ausgabe import kopf, buch

rng = np.random.default_rng(410)

kopf("Satz 4.6 -- Inversion: Exponentialverteilung aus Gleichverteilten")
x = -np.log(1 - rng.random(100_000)) / 2.0
print(f"Mittel {x.mean():.4f} (1/lambda = 0,5), Anteil über 1: {np.mean(x > 1):.4f} (e^-2 = {math.exp(-2):.4f})")
buch("Band 4, 4.4 'Warum jede Simulation damit anfängt'; ausgeführt in Band 27, Satz 2.4")

kopf("Kapitel 10 -- Monte Carlo: E[g(X)] für g(x) = x^4, X standardnormal (exakt 3)")
z = np.concatenate(box_muller(rng, 50_000))
print(f"Schätzung {np.mean(z ** 4):.3f} +- {np.std(z ** 4) / math.sqrt(len(z)):.3f}")
buch("die Genauigkeit wächst mit 1/sqrt(n), unabhängig von der Dimension -- Band 27, Satz 1.2")

kopf("Kapitel 12 -- Markow-Ketten-Monte-Carlo: Ziehen aus pi(x) ~ exp(-x^4)")
k, acc = metropolis_zufallsweg(lambda x: -x[0] ** 4, (0.0,), 1.0, 100_000, rng)
print(f"Annahme {acc:.0%}, Mittel {k.mean():+.3f} (exakt 0), E[x^2] {np.mean(k ** 2):.3f} (exakt {math.gamma(0.75) / math.gamma(0.25):.3f})")
buch("Band 4, 12.4; ausgeführt in Band 27, Kapitel 7")
