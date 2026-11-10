"""Band 18, Kapitel 3 -- Diskrete Dynamik (deterministisch, kein Zufall).

Erzeugt: die logistische Abbildung x -> a x (1 - x) mit a = 3,9 von zwei Startwerten,
die sich um ein Promille unterscheiden (3.5), und den Lyapunow-Exponenten.
"""
import _pfad
import math
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

a = 3.9
def bahn(x0, n):
    x = np.empty(n + 1); x[0] = x0
    for k in range(n): x[k + 1] = a * x[k] * (1 - x[k])
    return x

kopf("Zwei Bahnen, Startwerte 0,400 und 0,401 (3.5)")
b1, b2 = bahn(0.400, 40), bahn(0.401, 40)
zeile("Schritt", "x (0,400)", "x (0,401)", "Abstand", breite=12)
for k in (0, 5, 10, 14, 20, 30):
    zeile(k, round(b1[k], 4), round(b2[k], 4), f"{abs(b1[k] - b2[k]):.1e}", breite=12)
buch("nach etwa 14 Schritten ist die Vorhersage wertlos: ln(1000)/0,5 = 13,8")

kopf("Der Lyapunow-Exponent: Mittel von ln|a (1 - 2x)| entlang der Bahn")
x = bahn(0.4, 100_000)[1000:]
lam = np.mean(np.log(np.abs(a * (1 - 2 * x))))
print(f"lambda = {lam:.3f}; Fehler wächst wie e^(lambda n): ein Promille wird nach {math.log(1000) / lam:.1f} Schritten zu eins")
buch("lambda etwa 0,5 für a = 3,9")
