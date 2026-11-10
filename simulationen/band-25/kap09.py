"""Band 25, Kapitel 9 -- Konvergenz von Martingalen.

Erzeugt: die Pólya-Urne (9.2) -- der Grenzanteil ist gleichverteilt -- und die
Ungleichung von Doob gegen das Spiegelungsprinzip an der Irrfahrt (Satz 9.2).
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm
from mvg.prozesse import polya_urne, irrfahrten
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2509)

kopf("Die Pólya-Urne: 20 000 Urnen mit je 2000 Ziehungen, Start mit je einer Kugel (9.2)")
anteil = polya_urne(rng, 20_000, 2000)
zeile("Endanteil rot", "simuliert", "gleichverteilt", breite=16)
for a in np.arange(0, 1, 0.2):
    zeile(f"{a:.1f}-{a + .2:.1f}", round(float(((anteil >= a) & (anteil < a + .2)).mean()), 3), 0.2, breite=16)
print(f"Standardabweichung des Endanteils {anteil.std():.3f} (Gleichverteilung: {1 / math.sqrt(12):.3f})")
buch("fünf Fünftel von je 0,20, Abweichung höchstens 0,002; Streuung 0,289")

kopf("Satz 9.2 -- Doob gegen Spiegelung: P(max S_k >= a) bei 100 Schritten, 20 000 Läufe")
S = irrfahrten(rng, 20_000, 100)[:, :, 0]
m = S.max(axis=1)
zeile("a", "simuliert", "Doob", "Spiegelung", breite=12)
for a in (10, 15, 20):
    spiegel = 2 * norm.sf(a / 10)                            # Spiegelungsprinzip: P(max S >= a) = 2 P(S_n >= a)
    zeile(a, round(float((m >= a).mean()), 3), round(min(1.0, 100 / a ** 2), 3), round(spiegel, 3), breite=12)
buch("0,315 / 1,000 / 0,317; 0,132 / 0,444 / 0,134; 0,045 / 0,250 / 0,046")
