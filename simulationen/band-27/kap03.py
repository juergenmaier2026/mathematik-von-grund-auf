"""Band 27, Kapitel 3 -- Der Fluch der Dimension.

Erzeugt: Trapezregel gegen Monte Carlo in einer Dimension (Satz 3.1), die
Tabelle für d = 1, 3, 6, 10 (3.1), das Kugelvolumen (3.2) und den Bruchfall
mit unendlicher Varianz (3.3).
"""
import _pfad
import math
import numpy as np
from scipy.special import erf, gamma
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2703)

def f1(x):
    return np.exp(-x ** 2)

exakt1 = math.sqrt(math.pi) / 2 * erf(1.0)          # int_0^1 e^{-x^2} dx = 0,74682

kopf("Eine Dimension: Trapezregel gegen Monte Carlo (Satz 3.1)")
zeile("n", "Trapez-Fehler", "MC-Fehler (Mittel aus 200)", breite=28)
for n in (10, 100, 1000):
    x = np.linspace(0, 1, n + 1)
    trapez = np.trapezoid(f1(x), x)
    mc = np.array([abs(f1(rng.random(n)).mean() - exakt1) for _ in range(200)])
    zeile(n, f"{abs(trapez - exakt1):.0e}", f"{np.sqrt((mc ** 2).mean()):.0e}", breite=28)
buch("3.1: Trapez 6e-4, 6e-6, 6e-8; Monte Carlo 9e-2, 8e-3, 3e-3")

kopf("Integral von exp(-|x|^2) über [0,1]^d mit je 100 000 Auswertungen")
zeile("d", "exakt", "Monte Carlo", "+-", "Trapez", "je Achse", breite=12)
for d in (1, 3, 6, 10):
    exakt = exakt1 ** d
    N = 100_000
    y = np.exp(-(rng.random((N, d)) ** 2).sum(axis=1))
    mc, se = y.mean(), y.std(ddof=1) / math.sqrt(N)
    m = max(2, int(round(N ** (1 / d))))                 # Punkte je Achse
    x = np.linspace(0, 1, m)
    w = np.full(m, 1.0 / (m - 1)); w[0] = w[-1] = 0.5 / (m - 1)
    g = f1(x) * w                                        # eindimensionale Trapezgewichte
    trapez = g.sum() ** d                                # Produktregel: Produkt der 1-D-Summen
    zeile(d, round(exakt, 5), round(mc, 4), round(se, 4), round(trapez, 5), m, breite=12)
buch("Tabelle in 3.1: ab d = 6 liegt die Trapezregel außerhalb jedes Fehlerbalkens")

kopf("Das Kugelvolumen aus 200 000 Punkten im Würfel (Abschnitt 3.2)")
zeile("d", "exakt", "geschätzt", "Trefferquote", "rel. Fehler", breite=14)
for d in (2, 5, 10, 20):
    exakt = math.pi ** (d / 2) / gamma(d / 2 + 1)
    N = 200_000
    treffer = ((rng.uniform(-1, 1, (N, d)) ** 2).sum(axis=1) <= 1).mean()
    schaetz = treffer * 2 ** d
    relfehler = math.sqrt((1 - treffer) / (N * treffer)) if treffer > 0 else float("nan")
    zeile(d, round(exakt, 4), round(schaetz, 4), f"{treffer:.2e}", f"{relfehler:.1%}" if treffer else "---", breite=14)
buch("3.2: in 20 Dimensionen kein einziger Treffer -- exakt 0,0258, Trefferquote 2e-8")

kopf("Der Bruchfall: int_0^1 x^(-1/2) dx = 2 mit unendlicher Varianz (3.3)")
zeile("n", "Schätzung", "geschätzter SE", breite=16)
for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
    y = rng.random(n) ** -0.5
    zeile(n, round(y.mean(), 4), round(y.std(ddof=1) / math.sqrt(n), 4), breite=16)
t = rng.random(10 ** 5)
y = 2 * np.ones_like(t)                                   # Substitution x = t^2: Integrand 2 t / t = 2
print(f"Nach Substitution x = t^2: Schätzung {y.mean():.4f}, Varianz {y.var():.1f}")
buch("3.3: der Schätzer konvergiert, aber der Standardfehler springt; nach Substitution Varianz null")
