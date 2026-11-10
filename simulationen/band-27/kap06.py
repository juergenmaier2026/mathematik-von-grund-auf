"""Band 27, Kapitel 6 -- Dort ziehen, wo etwas passiert.

Erzeugt: die seltenen Abweisungstage (6.1), die Verschiebung für P(Z > 4)
(6.3) mit wirksamem n, Quasi-Monte-Carlo gegen Monte Carlo (6.4), den Bruchfall
mit zu dünnen Rändern (6.5) und die exakte Rechnung aus Aufgabe 6.1.
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import minimize_scalar
from mvg.station import Intensivstation
from mvg.mcmc import wichtigkeit
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2706)

kopf("Wie oft kommt es ganz schlimm? Abweisungen je Tag aus 400 000 simulierten Tagen (6.1)")
r = Intensivstation(12).lauf(400_000, einschwingen=100, seed=2706, protokoll=True)
je_tag = r["abweisungen_je_tag"]
zeile("mindestens", "Anteil der Tage", "etwa alle", breite=16)
for k in (1, 3, 5, 7):
    p = (je_tag >= k).mean()
    zeile(k, f"{p:.5f}", f"{1 / p:.0f} Tage" if p else "nie gesehen", breite=16)
buch("0,235 / 0,050 / 0,0068 / 0,00057 -- für sieben Abweisungen 226 Treffer")

kopf("P(Z > 4) durch Verschiebung des Vorschlags auf N(mu, 1) (6.3)")
exakt = norm.sf(4)
N, R = 100_000, 100
print(f"exakt {exakt:.3e}; naiv erwartet n p = {N * exakt:.1f} Treffer je {N} Ziehungen")
zeile("mu", "Schätzung", "rel. Fehler", "Varianzfaktor", "n_eff", breite=14)
var_naiv = exakt * (1 - exakt) / N
for mu in (0, 2, 4, 6, 8):
    erg = []
    for _ in range(R):
        sch, se, neff = wichtigkeit(lambda x: (x > 4).astype(float),
                                    lambda x: norm.logpdf(x), lambda x: norm.logpdf(x, mu),
                                    lambda r, n: r.normal(mu, 1, n), N, rng)
        erg.append((sch, neff))
    sch = np.array([e[0] for e in erg])
    zeile(mu, f"{sch.mean():.3e}", f"{sch.std(ddof=1) / exakt:.1%}", round(var_naiv / sch.var(ddof=1), 0), round(np.mean([e[1] for e in erg]), 0))
buch("Faktoren 1 / 550 / 6 700 / 900 / 3; n_eff bei mu = 4 etwa 18 000")

kopf("Aufgabe 6.1 -- zweites Moment e^(mu^2) P(Z > 4 + mu) exakt")
zeile("mu", "2. Moment", "Varianz", breite=14)
for mu in (0, 2, 4, 6, 8):
    m2 = math.exp(mu * mu) * norm.sf(4 + mu)
    zeile(mu, f"{m2:.2e}", f"{m2 - exakt ** 2:.2e}")
opt = minimize_scalar(lambda m: m * m + norm.logsf(4 + m), bounds=(0, 8), method="bounded")
print(f"optimales mu = {opt.x:.2f}; Varianzfaktor gegen naiv bei mu = 4: {exakt / (math.exp(16) * norm.sf(8) - exakt ** 2):.0f}")
buch("mu* = 4,1, Faktor 7 000")

kopf("Quasi-Monte-Carlo: Integral von exp(x_1 + ... + x_d) / (e-1)^d = 1, n = 16 384 (6.4)")
n = 2 ** 14
zeile("d", "Monte Carlo", "Sobol (verwürfelt)", "Faktor", breite=20)
for d in (2, 5, 10, 20):
    f = lambda X: np.exp(X.sum(axis=1)) / (math.e - 1) ** d
    mc = np.array([f(rng.random((n, d))).mean() - 1 for _ in range(20)])
    qm = np.array([f(qmc.Sobol(d, scramble=True, seed=int(rng.integers(1e9))).random(n)).mean() - 1 for _ in range(20)])
    rm, rq = np.sqrt((mc ** 2).mean()), np.sqrt((qm ** 2).mean())
    zeile(d, f"{rm:.1e}", f"{rq:.1e}", f"{rm / rq:,.0f}", breite=20)
buch("Faktoren etwa 24 000 / 740 / 38 / 4 -- in kleiner Dimension schwankt der Faktor stark mit der Verwürfelung")

kopf("Der Bruchfall: E[X^2] = 1 für N(0,1) mit Vorschlag N(0, 0,6^2) -- unendliche Varianz (6.5)")
s = 0.6
zeile("n", "Schätzung", "n_eff", "größtes Gewicht", breite=16)
for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
    x = rng.normal(0, s, n)
    w = np.exp(norm.logpdf(x) - norm.logpdf(x, 0, s))
    zeile(n, round(float(np.mean(x * x * w)), 3), round(w.sum() ** 2 / np.dot(w, w), 0), f"{w.max() / w.sum():.2%}", breite=16)
print(f"f^2/g ~ exp(x^2 (1/(2 s^2) - 1)) = exp({1 / (2 * s * s) - 1:.2f} x^2) -> unendliches zweites Moment")
buch("Schätzungen 0,79 / 1,31 / 1,04 / 0,98 -- zwischen 10^3 und 10^4 wird es schlechter")
