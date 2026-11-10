"""Band 26, Kapitel 8 -- Dynamische Panels.

Erzeugt: die Nickell-Verzerrung (Satz 8.2) für rho = 0,5 und T = 3, 5, 10, 30 bei
5000 Personen -- Näherung gegen Simulation (Aufgabe 8.1).
"""
import _pfad
import numpy as np
from mvg.oekonometrie import steigung, innerhalb
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2608)
rho, n = 0.5, 5000
kopf("Nickell: feste Effekte mit verzögerter Zielgröße, rho = 0,5 (Aufgabe 8.1)")
zeile("T", "Näherung", "simuliert", "gepoolt", breite=12)
for T in (3, 5, 10, 30):
    alpha = rng.standard_normal(n)
    y = np.empty((n, T + 50))
    y[:, 0] = alpha / (1 - rho)
    for t in range(1, T + 50):
        y[:, t] = alpha + rho * y[:, t - 1] + rng.standard_normal(n)
    y = y[:, 50:]                                                      # Einschwingphase verworfen
    yl, yn = innerhalb(y[:, :-1]), innerhalb(y[:, 1:])
    zeile(T, round(rho - (1 + rho) / (T - 1), 3), round(steigung(yn.ravel(), yl.ravel()), 3), round(steigung(y[:, 1:].ravel(), y[:, :-1].ravel()), 3), breite=12)
buch("-0,250 / 0,125 / 0,333 / 0,448 gegen -0,073 / 0,158 / 0,337 / 0,449 -- die Näherung gilt für großes T; "
     "bei kleinem T hängt die Verzerrung vom Anfangszustand ab (hier stationär gestartet, im Buch nicht)")
