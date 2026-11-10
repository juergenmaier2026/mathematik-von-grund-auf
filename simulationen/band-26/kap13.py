"""Band 26, Kapitel 13 -- Politik.

Erzeugt: die Lucas-Kritik im Leitfall (13.1): y = 8 + beta z + gamma (z^e - z) + e
mit beta = -0,10, gamma = 0,05; geschätzt wird beta + gamma (phi - 1), T = 20 000.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import steigung
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2613)
beta, gamma, T = -0.10, 0.05, 20_000
kopf("Lucas: der Koeffizient hängt von der Regel ab (13.1)")
zeile("Regel", "phi", "Formel", "simuliert", breite=18)
for name, phi in (("träge Regel", 0.8), ("sofortige Rückkehr", 0.0)):
    z = np.empty(T + 1); z[0] = 10
    for t in range(T):
        z[t + 1] = 10 + phi * (z[t] - 10) + rng.standard_normal()
    ze = 10 + phi * (z[:-1] - 10)                                   # Erwartung für t+1 aus Sicht von t
    y = 8 + beta * z[:-1] + gamma * (ze - z[:-1]) + rng.normal(0, 0.5, T)
    zeile(name, phi, round(beta + gamma * (phi - 1), 3), round(steigung(y, z[:-1]), 3), breite=18)
buch("-0,110 / -0,103 und -0,150 / -0,141")
