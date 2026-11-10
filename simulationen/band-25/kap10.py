"""Band 25, Kapitel 10 -- Die Brownsche Bewegung.

Erzeugt: quadratische und totale Variation (Satz 10.2), den Differenzenquotienten
(Satz 10.3), das Maximum (Spiegelung, 10.3) und die Nullstellen der Irrfahrt (10.4).
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm
from mvg.prozesse import brownsche_pfade, irrfahrten
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2510)

kopf("Quadratische und totale Variation eines Pfads auf [0, 1] (Satz 10.2)")
zeile("Schritte", "quadratische", "totale", breite=14)
for n in (100, 1000, 10_000, 100_000):
    W = brownsche_pfade(rng, 1, n)[0]; dW = np.diff(W)
    zeile(n, round(float((dW ** 2).sum()), 3), round(float(np.abs(dW).sum()), 1), breite=14)
buch("quadratische Variation gegen 1, totale gegen unendlich (0,776 / 0,985 / 0,989 / 1,008 und 6,8 / 25,2 / 79,2 / 253,2)")

kopf("Satz 10.3 -- der mittlere Betrag eines Zuwachses, geteilt durch sqrt(dt)")
dW = np.diff(brownsche_pfade(rng, 100, 10_000), axis=1)
print(f"simuliert {np.abs(dW).mean() * math.sqrt(10_000):.4f}, theoretisch sqrt(2/pi) = {math.sqrt(2 / math.pi):.4f}")
buch("0,8006 gegen 0,7979")

kopf("Das Maximum: P(W_1 >= 1) und P(max W >= 1) an 100 000 Pfaden mit 1000 Schritten (10.3)")
W = brownsche_pfade(rng, 100_000, 1000)
print(f"P(W_1 >= 1): theoretisch {norm.sf(1):.4f}, simuliert {float((W[:, -1] >= 1).mean()):.4f}")
print(f"P(max >= 1): theoretisch {2 * norm.sf(1):.4f}, simuliert {float((W.max(axis=1) >= 1).mean()):.4f}")
buch("0,1587 und 0,3173 / 0,3182 -- das Gitter unterschätzt das Maximum ein wenig")

kopf("Nullstellen der Irrfahrt (10.4), je 2000 Läufe")
zeile("n", "sqrt(2n/pi)", "simuliert", breite=14)
for n in (1000, 10_000, 100_000):
    S = irrfahrten(rng, 2000, n)[:, 1:, 0]
    zeile(n, round(math.sqrt(2 * n / math.pi), 1), round(float((S == 0).sum(axis=1).mean()), 1), breite=14)
buch("25,2 / 79,2 / 253,2")
