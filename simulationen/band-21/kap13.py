"""Band 21, Kapitel 13 -- Auktionen.

Erzeugt: Erlösäquivalenz von Erst- und Zweitpreisauktion bei n Bietern mit
gleichverteilten Werten (13.4, 200 000 Durchläufe) und den Fluch des Gewinners (13.5).
"""
import _pfad
import math
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2113)

kopf("Erlös bei n Bietern, Werte gleichverteilt auf [0, 1], 200 000 Auktionen (13.4)")
zeile("n", "Formel (n-1)/(n+1)", "Zweitpreis", "Erstpreis", breite=18)
for n in (2, 3, 5, 10):
    v = rng.random((200_000, n)); v.sort(axis=1)
    zweit = v[:, -2].mean()                                   # zweithöchster Wert
    erst = ((n - 1) / n * v[:, -1]).mean()                    # Gleichgewichtsgebot (n-1)/n v des Höchsten
    zeile(n, round((n - 1) / (n + 1), 4), round(float(zweit), 4), round(float(erst), 4), breite=18)
buch("bei n = 3: 0,5002 und 0,4997")

kopf("Der Fluch des Gewinners: wahrer Wert 100, Schätzfehler normalverteilt mit Streuung 10 (13.5)")
zeile("n", "Höchstschätzung", "Überschätzung", "Näherung 10 sqrt(2 ln n)", breite=22)
for n in (2, 5, 10, 20):
    hoechst = (100 + rng.normal(0, 10, (200_000, n))).max(axis=1).mean()
    zeile(n, round(float(hoechst), 1), f"{hoechst - 100:+.1f}", round(10 * math.sqrt(2 * math.log(n)), 1), breite=22)
buch("105,7 / 111,5 / 115,4 / 118,7 -- die Näherung 17,9 bei n = 5 taugt erst für große n")
