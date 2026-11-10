"""Band 25, Kapitel 8 -- Martingale.

Erzeugt: die Wartezeit auf Münzwurfmuster (8.4): KZK braucht im Mittel zehn,
KKK vierzehn Würfe -- der Satz von Conway, geprüft an 20 000 Läufen.
"""
import _pfad
import numpy as np
from mvg.prozesse import wartezeit_muster
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2508)

kopf("Wartezeit auf ein Muster, je 20 000 Läufe (8.4)")
zeile("Muster", "erwartet", "simuliert", breite=12)
for muster, erwartet in (("KZK", 10), ("KKK", 14), ("KKZ", 8), ("KZZ", 8)):
    zeile(muster, erwartet, round(wartezeit_muster(rng, muster, 20_000), 2), breite=12)
buch("9,92 und 13,98 -- die Überlappung des Musters mit sich selbst bestimmt die Wartezeit")
