"""Band 25, Kapitel 1 -- Die Irrfahrt.

Erzeugt: das Arkussinus-Gesetz und die Vorzeichenwechsel an 20 000 Läufen
(1.2) und die Rückkehr in einer, zwei und drei Dimensionen (Satz 1.4).
"""
import _pfad
import math
import numpy as np
from mvg.prozesse import irrfahrten
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2501)

kopf("Arkussinus-Gesetz: Zeitanteil in Führung, 20 000 Läufe à 2000 Schritte (1.2)")
S = irrfahrten(rng, 20_000, 2000)[:, :, 0]
oben = ((S[:, 1:] > 0) | ((S[:, 1:] == 0) & (S[:, :-1] > 0))).mean(axis=1)   # Zeitanteil "oben"
arcsin = lambda a, b: 2 / math.pi * (math.asin(math.sqrt(b)) - math.asin(math.sqrt(a)))
zeile("Zeitanteil oben", "simuliert", "theoretisch", breite=18)
for name, a, b in (("unter 10 %", 0, .1), ("10-20 %", .1, .2), ("40-50 %", .4, .5)):
    zeile(name, round(float(((oben >= a) & (oben < b)).mean()), 3), round(arcsin(a, b), 3), breite=18)
zeile("<10 % oder >90 %", round(float(((oben < .1) | (oben > .9)).mean()), 3), round(2 * arcsin(0, .1), 3), breite=18)
zeile("40-60 %", round(float(((oben >= .4) & (oben < .6)).mean()), 3), round(arcsin(.4, .6), 3), breite=18)
null = (S[:, 1:] == 0).sum(axis=1).mean()
print(f"Nullstellen in 2000 Schritten: simuliert {null:.1f}, sqrt(2n/pi) = {math.sqrt(2 * 2000 / math.pi):.1f}")
buch("0,214/0,205; 0,090/0,090; 0,062/0,064; 0,408/0,410; 0,128/0,128; Nullstellen 34,8 / 35,7")

kopf("Satz 1.4 -- Rückkehr zum Ausgangspunkt innerhalb von 5000 Schritten")
zeile("d", "Rückkehr", "Grenzwert", breite=12)
for d, grenz in ((1, 1.0), (2, 1.0), (3, 0.3405)):
    P = irrfahrten(rng, 4000, 5000, d)
    zurueck = (np.abs(P[:, 1:, :]).sum(axis=2) == 0).any(axis=1).mean()
    zeile(d, round(float(zurueck), 3), grenz, breite=12)
buch("0,990 / 0,726 / 0,332 -- in drei Dimensionen 0,3405 (Pólya)")
