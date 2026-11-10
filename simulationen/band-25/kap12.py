"""Band 25, Kapitel 12 -- Die Itô-Formel.

Erzeugt: Erwartungswert gegen Median der geometrischen Brownschen Bewegung
(12.2, 200 000 Pfade) und den Ornstein-Uhlenbeck-Prozess (12.4).
"""
import _pfad
import math
import numpy as np
from mvg.prozesse import ornstein_uhlenbeck
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2512)

kopf("Geometrische Brownsche Bewegung S0 = 1, mu = 0,08, sigma = 0,20 (12.2)")
mu, sg = 0.08, 0.20
zeile("Jahre", "Erwartung", "Median", "Verhältnis", "sim. Mittel", "sim. Median", breite=12)
for T in (1, 10, 30):
    S = np.exp((mu - sg ** 2 / 2) * T + sg * math.sqrt(T) * rng.standard_normal(200_000))
    zeile(T, round(math.exp(mu * T), 3), round(math.exp((mu - sg ** 2 / 2) * T), 3), round(math.exp(-sg ** 2 / 2 * T), 3),
          round(float(S.mean()), 2), round(float(np.median(S)), 2), breite=12)
buch("30 Jahre: 11,02 gegen 6,05; simuliert 11,01 und 6,02")

kopf("Ornstein-Uhlenbeck: theta = 2, xbar = 1, sigma = 0,5, 200 000 Schritte mit dt = 0,01 (12.4)")
x = ornstein_uhlenbeck(rng, 2.0, 1.0, 0.5, 1.0, 0.01, 200_000)
print(f"Mittelwert theoretisch 1,00, simuliert {x.mean():.2f}; Streuung sigma/sqrt(2 theta) = {0.5 / math.sqrt(4):.3f}, simuliert {x.std():.3f}")
buch("0,99 und 0,232 -- die Streuung hängt von dt ab: Euler-Maruyama unterschätzt sie um etwa theta dt / 2")
