"""Band 27, Kapitel 2 -- Zufall aus einer Formel.

Erzeugt: die Beispiele mit m = 16 (2.1), den RANDU-Test (Satz 2.3), die
Inversion (Satz 2.4, Aufgabe 2.2), Box-Muller (Satz 2.5), die Verwerfung
(Satz 2.6, Aufgabe 2.3) und die Prüfzahlen aus Aufgabe 2.1.
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm
from mvg.zufall import lcg, randu, box_muller, verwerfung
from mvg.ausgabe import kopf, buch

kopf("Zwei Generatoren mit m = 16 (Abschnitt 2.1)")
print("a=5, c=1:", lcg(5, 1, 16, 0, 18))
print("a=5, c=0:", lcg(5, 0, 16, 1, 8))
buch("2.1: volle Periode 16 bzw. Periode 4")

kopf("RANDU und der Ebenentest (Satz 2.3)")
u = randu(1, 30_002)
print(f"Mittel {u.mean():.3f}, Varianz {u.var():.4f} (gleichverteilt: 0,5 und 0,0833),"
      f" Autokorrelation Abstand 1: {np.corrcoef(u[:-1], u[1:])[0, 1]:+.4f}")
rng = np.random.default_rng(2702)
v = rng.random(30_002)
for name, x in (("RANDU", u), ("PCG64", v)):
    t = (9 * x[:-2] - 6 * x[1:-1] + x[2:]) % 1.0
    verschieden = len(np.unique(np.round(t, 9)))
    print(f"{name:6s}: (9u_n - 6u_n+1 + u_n+2) mod 1 nimmt {verschieden} verschiedene Werte an ({len(t)} Tripel)")
buch("RANDU ein Wert, PCG64 etwa 29 540 verschiedene -- alle RANDU-Tripel liegen auf 15 Ebenen")

kopf("Inversion (Satz 2.4)")
u = rng.random(100_000)
x = -np.log(1 - u) / 2.4
print(f"Exponential(2,4) durch Inversion: Mittel {x.mean():.4f} bei 1/lambda = {1 / 2.4:.4f}")
d = 4 * math.exp(0.6 * norm.ppf(0.37))
print(f"Aufgabe 2.2: u = 0,37 -> Phi^-1 = {norm.ppf(0.37):.3f} -> Verweildauer {d:.2f} Tage;"
      f" Probe Phi(ln(d/4)/0,6) = {norm.cdf(math.log(d / 4) / 0.6):.3f}")
buch("Mittel 0,4175 bei 0,4167; Aufgabe 2.2: 3,28 Tage")

kopf("Box-Muller (Satz 2.5)")
z1, z2 = box_muller(rng, 100_000)
z = np.concatenate([z1, z2])
schiefe = np.mean(z ** 3) / np.std(z) ** 3
print(f"100 000 Paare: Mittel {z.mean():+.4f}, Streuung {z.std():.3f}, Schiefe {schiefe:+.3f}")
buch("Mittel -0,0007, Streuung 0,998, Schiefe 0,001")

kopf("Verwerfung (Satz 2.6, Aufgabe 2.3)")
# Halbkreisdichte f(x) = (2/pi) sqrt(1-x^2) auf [-1,1] aus der Gleichverteilung: M = 4/pi
_, quote = verwerfung(rng, lambda r, k: r.uniform(-1, 1, k), lambda x: np.sqrt(1 - x ** 2), 100_000)
print(f"Halbkreis aus Rechteck:     1/M = {math.pi / 4:.3f}, Annahmequote {quote:.3f}")
# Halbnormalverteilung aus Exponential(1): f/(M g) = exp(-(x-1)^2/2), M = sqrt(2e/pi)
_, quote = verwerfung(rng, lambda r, k: r.exponential(1.0, k), lambda x: np.exp(-(x - 1) ** 2 / 2), 100_000)
print(f"Halbnormal aus Exponential: 1/M = {math.sqrt(math.pi / (2 * math.e)):.3f}, Annahmequote {quote:.3f}")
for lam in (0.5, 1.0, 2.0):
    M = math.sqrt(2 / math.pi) / lam * math.exp(lam ** 2 / 2)
    print(f"   Aufgabe 2.3, Hülle Exponential({lam}): M = {M:.2f}, Annahme {1 / M:.0%}")
buch("0,785/0,787 und 0,760/0,761; Aufgabe 2.3: lambda = 1 ist die beste Hülle")

kopf("Aufgabe 2.1 -- Einen Generator prüfen")
x = rng.random(100_000)
klassen = np.histogram(x, bins=100, range=(0, 1))[0]
chi2 = ((klassen - 1000) ** 2 / 1000).sum()
print(f"chi^2 mit 99 Freiheitsgraden: {chi2:.1f} (Erwartung 99, Streuung {math.sqrt(2 * 99):.1f});"
      f" Autokorrelation Abstand 1: {np.corrcoef(x[:-1], x[1:])[0, 1]:+.4f} (Standardfehler {1 / math.sqrt(100_000):.4f})")
buch("Aufgabe 2.1: 105,5 und +0,003 sind unauffällig")
