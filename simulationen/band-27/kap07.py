"""Band 27, Kapitel 7 -- Ziehen, ohne ziehen zu können.

Erzeugt: die Schrittweitentabelle des Zufallsweg-Metropolis (7.3), den
Gibbs-Sampler (7.4), die abgeleitete Größe (7.5), den Bruchfall mit zwei
Gipfeln (7.6) und den Metropolis-Schritt von Hand (Aufgabe 7.1).
"""
import _pfad
import math
import numpy as np
from mvg.mcmc import metropolis_zufallsweg, gibbs_normal
from mvg.statistik import tau_int
from mvg.ausgabe import kopf, zeile, buch

# Die Daten: hundert logarithmierte Aufenthalte -- derselbe Startwert wie im Buch
rng = np.random.default_rng(7)
y = rng.normal(math.log(4), 0.6, 100)
n, ybar, s = len(y), y.mean(), y.std(ddof=1)
print(f"Daten: n = {n}, ybar = {ybar:.3f}, s = {s:.3f}")
buch("7.1: ybar 1,283, s 0,528")

def log_post(theta):                       # theta = (mu, log sigma); flache A-priori-Dichte
    mu, ls = theta
    sg = math.exp(ls)
    return -n * ls - np.sum((y - mu) ** 2) / (2 * sg * sg)

kopf("Zufallsweg-Metropolis für (mu, log sigma), 20 000 Schritte (7.3)")
zeile("Schritt", "Annahme", "tau", "wirksames n", "Streuung mu", breite=13)
for schritt in (0.01, 0.05, 0.10, 0.20, 1.0):
    kette, acc = metropolis_zufallsweg(log_post, (ybar, math.log(s)), schritt, 20_000, rng)
    mu = kette[2000:, 0]
    tau = tau_int(mu, max_lag=2000)
    zeile(schritt, f"{acc:.1%}", round(tau, 1), round(len(mu) / tau, 0), round(mu.std(), 4), breite=13)
print(f"exakt (t-Verteilung): Streuung von mu = {s / math.sqrt(n) * math.sqrt((n - 1) / (n - 3)):.4f}")
buch("Annahme 91/61/36/15/0,8 %; tau 108/10/6,4/11/184; Streuung 0,053 in jeder Zeile")

kopf("Der Gibbs-Sampler, 20 000 Schritte (7.4)")
kette = gibbs_normal(y, 20_000, rng)
for j, name in enumerate(("mu", "sigma")):
    zeile(name, round(kette[:, j].mean(), 4), round(kette[:, j].std(), 4), round(tau_int(kette[:, j]), 1), breite=12)
buch("tau = 1,0: so gut wie unabhängige Ziehungen")

kopf("Abgeleitete Größe: mittlere Verweildauer exp(mu + sigma^2/2) (7.5)")
m = np.exp(kette[:, 0] + kette[:, 1] ** 2 / 2)
print(f"A-posteriori-Mittel {m.mean():.2f} Tage, 95-%-Intervall [{np.percentile(m, 2.5):.2f}; {np.percentile(m, 97.5):.2f}], P(Mittel > 5) = {np.mean(m > 5):.4f}")
buch("4,16 Tage, [3,74; 4,67], 0,0012")

kopf("Der Bruchfall: zwei Gipfel bei -6 und +6, Schrittweite 1 gegen 6 (7.6)")
def log_bimodal(x):
    return np.logaddexp(-(x[0] + 6) ** 2 / 2, -(x[0] - 6) ** 2 / 2)
for schritt in (1.0, 6.0):
    print(f"Schrittweite {schritt}:")
    zeile("Start", "Mittel", "Annahme", "tau", breite=10)
    for start in (-6, -5, 5, 6):
        k, acc = metropolis_zufallsweg(log_bimodal, (start,), schritt, 5000, rng)
        zeile(start, round(k[:, 0].mean(), 2), f"{acc:.0%}", round(tau_int(k[:, 0], max_lag=500), 1), breite=10)
buch("Schritt 1: jede Kette makellos (71 %, tau 8) und um 6 falsch; Schritt 6: alle Mittel nahe 0, Annahme 24 %")

kopf("Aufgabe 7.1 -- ein Metropolis-Schritt von Hand, pi ~ exp(-x^2/2)")
for xt, xn, u in ((1.0, 1.8, 0.35), (1.0, 0.4, 0.35)):
    q = math.exp(-(xn * xn - xt * xt) / 2)
    print(f"x_t = {xt}, x' = {xn}: Quotient {q:.3f}, u = {u} -> {'angenommen' if u < q else 'abgelehnt'}")
buch("0,326 -> abgelehnt; 1,52 -> angenommen ohne Ziehung")
