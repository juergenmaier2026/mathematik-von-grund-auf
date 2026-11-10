"""Band 27, Kapitel 5 -- Weniger Streuung ohne mehr Läufe.

Erzeugt: antithetische Variablen (Satz 5.1), Kontrollvariablen für das
Integral und für die Intensivstation (Satz 5.2), Konditionierung (Satz 5.3),
den Bruchfall (5.6) und die Rechnung aus Aufgabe 5.2.
"""
import _pfad
import math
import numpy as np
from mvg.station import Intensivstation
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2705)
N, R = 10_000, 200                         # Ziehungen je Schätzung, Wiederholungen

kopf("Satz 5.1 -- antithetische Paare für int_0^1 e^x dx = e - 1")
naiv = np.array([np.exp(rng.random(N)).mean() for _ in range(R)])
anti = np.array([((lambda u: (np.exp(u) + np.exp(1 - u)) / 2)(rng.random(N // 2))).mean() for _ in range(R)])
u = rng.random(10 ** 6)
rho = np.corrcoef(np.exp(u), np.exp(1 - u))[0, 1]
zeile("Verfahren", "SE", "Varianzfaktor", breite=14)
zeile("naiv", round(naiv.std(ddof=1), 4), 1.0)
zeile("antithetisch", round(anti.std(ddof=1), 4), round(naiv.var(ddof=1) / anti.var(ddof=1), 1))
print(f"Korrelation von e^U und e^(1-U): {rho:.3f};  1 + rho = {1 + rho:.3f} = 1/{1 / (1 + rho):.0f}")
buch("naiv 0,0048, antithetisch 0,0009, Faktor 31; rho = -0,968")

kopf("Satz 5.2 -- Kontrollvariable U für dasselbe Integral")
kv = []
for _ in range(R):
    u = rng.random(N); y = np.exp(u)
    beta = np.cov(y, u)[0, 1] / u.var(ddof=1)
    kv.append(y.mean() - beta * (u.mean() - 0.5))
kv = np.array(kv)
print(f"SE naiv {naiv.std(ddof=1):.4f}, mit Kontrolle {kv.std(ddof=1):.4f}, Faktor {naiv.var(ddof=1) / kv.var(ddof=1):.0f}")
buch("Faktor 56 (exakt 61, Aufgabe 5.2)")

kopf("Aufgabe 5.2 -- beta* und der Faktor exakt")
e = math.e
var_y = (e * e - 1) / 2 - (e - 1) ** 2
cov = 1 - (e - 1) / 2
rho = cov / math.sqrt(var_y / 12)
print(f"Var[e^U] = {var_y:.4f}, Cov = {cov:.4f}, beta* = {cov * 12:.3f}, rho = {rho:.4f}, Faktor 1/(1-rho^2) = {1 / (1 - rho ** 2):.0f}")
buch("beta* 1,691, rho 0,9918, Faktor 61")

kopf("Kontrollvariablen für die Intensivstation: 200 Läufe à 1000 Tage (50 Tage Einschwingphase)")
station = Intensivstation(12)
laeufe = station.wiederholungen(200, 1050, einschwingen=50, seed=2705)
p = np.array([r["anteil"] for r in laeufe])
C1 = np.array([r["ankuenfte"] for r in laeufe]) - 2.4 * 1000          # Zahl der Ankünfte, E = lambda T
C2 = np.array([r["mittlere_dauer"] for r in laeufe]) - station.mittlere_verweildauer
def mit_kontrolle(p, *C):
    X = np.column_stack(C)
    b = np.linalg.lstsq(np.column_stack([np.ones(len(p)), X]), p, rcond=None)[0][1:]
    return p - X @ b
zeile("Kontrolle", "rho", "Streuung", "Faktor", breite=22)
zeile("keine", "", round(p.std(ddof=1), 4), 1.0, breite=22)
for name, C in (("Zahl der Ankünfte", C1), ("mittlere Verweildauer", C2)):
    pc = mit_kontrolle(p, C)
    zeile(name, round(np.corrcoef(p, C)[0, 1], 2), round(pc.std(ddof=1), 4), round(p.var(ddof=1) / pc.var(ddof=1), 1), breite=22)
pc = mit_kontrolle(p, C1, C2)
zeile("beide", "", round(pc.std(ddof=1), 4), round(p.var(ddof=1) / pc.var(ddof=1), 1), breite=22)
buch("keine 0,0132; Ankünfte rho 0,72 Faktor 2,1; Verweildauer rho 0,52 Faktor 1,4; beide 0,0059 Faktor 5,0")

kopf("Satz 5.3 -- Konditionierung: P(X + Y > 4) für X, Y ~ Exp(1)")
exakt = 5 * math.exp(-4)
zaehl = np.array([(rng.exponential(1, N) + rng.exponential(1, N) > 4).mean() for _ in range(R)])
kond = np.array([np.exp(-np.maximum(4 - rng.exponential(1, N), 0)).mean() for _ in range(R)])
zeile("Verfahren", "Schätzung", "SE", "Faktor", breite=14)
zeile("zählen", round(zaehl.mean(), 4), round(zaehl.std(ddof=1), 4), 1.0)
zeile("konditionieren", round(kond.mean(), 4), round(kond.std(ddof=1), 4), round(zaehl.var(ddof=1) / kond.var(ddof=1), 1))
zeile("exakt 5e^-4", round(exakt, 4), "", "")
buch("0,0027 gegen 0,0016, Faktor 2,9")

kopf("Der Bruchfall: antithetisch bei g(u) = |u - 1/2|")
g = lambda u: np.abs(u - 0.5)
naiv = np.array([g(rng.random(N)).mean() for _ in range(R)])
anti = np.array([((g(u := rng.random(N // 2)) + g(1 - u)) / 2).mean() for _ in range(R)])
print(f"SE naiv {naiv.std(ddof=1):.5f}, antithetisch {anti.std(ddof=1):.5f} -- Varianz verdoppelt (Faktor {anti.var(ddof=1) / naiv.var(ddof=1):.2f})")
buch("5.6: g(U) = g(1-U), rho = +1, Varianz verdoppelt bei gleichem Aufwand")
