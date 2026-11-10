"""Band 25, Kapitel 4 -- Der Poisson-Prozess.

Erzeugt: das Wartezeitparadox (4.4) an 200 000 Ankünften.
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, buch

rng = np.random.default_rng(2504)

kopf("Das Wartezeitparadox: Ankünfte alle 18,75 Minuten im Mittel, 200 000 Ankünfte (4.4)")
abst = rng.exponential(18.75, 200_000)
t = np.cumsum(abst)
besucher = rng.uniform(0, t[-1], 200_000)                   # zufällige Zeitpunkte
idx = np.searchsorted(t, besucher)                           # nächste Ankunft nach dem Besucher
erlebt = abst[idx]
rest = t[idx] - besucher
print(f"mittlerer Abstand {abst.mean():.2f} min, vom Besucher erlebter Abstand {erlebt.mean():.2f} min, Restwartezeit {rest.mean():.2f} min")
buch("18,75 / 37,45 / 18,72 -- der Besucher fällt bevorzugt in lange Lücken")
