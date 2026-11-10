"""Band 27, Kapitel 8 -- Ist die Kette angekommen?

Erzeugt: R-Dach für die Ketten aus Kapitel 7 (Satz 8.2), das wirksame n
der Ketten (8.3), den Bruchfall mit vier Ketten in einem Gipfel (8.5),
R-Dach von Hand (Aufgabe 8.1) und die Planung aus Aufgabe 8.2.
"""
import _pfad
import math
import numpy as np
from mvg.mcmc import metropolis_zufallsweg, gibbs_normal
from mvg.statistik import r_dach, tau_int
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(7)
y = rng.normal(math.log(4), 0.6, 100)               # dieselben Daten wie in Kapitel 7
n, ybar, s = len(y), y.mean(), y.std(ddof=1)
rng = np.random.default_rng(2708)

def log_post(theta):
    mu, ls = theta
    return -n * ls - np.sum((y - mu) ** 2) / (2 * math.exp(2 * ls))

def log_bimodal(x):
    return np.logaddexp(-(x[0] + 6) ** 2 / 2, -(x[0] - 6) ** 2 / 2)

kopf("Satz 8.2 -- R-Dach für vier überstreute Ketten")
zeile("Ketten", "R-Dach", breite=34)
for name, ld, starts, schritt, laenge in (
        ("(mu,sigma), Schritt 0,05", log_post, [ybar - .5, ybar - .2, ybar + .2, ybar + .5], 0.05, 2000),
        ("(mu,sigma), Schritt 0,2", log_post, [ybar - .5, ybar - .2, ybar + .2, ybar + .5], 0.2, 2000),
        ("zwei Gipfel, Schritt 1", log_bimodal, [-6, -5, 5, 6], 1.0, 5000),
        ("zwei Gipfel, Schritt 6", log_bimodal, [-6, -5, 5, 6], 6.0, 5000)):
    ketten = []
    for st in starts:
        start = (st, math.log(s)) if ld is log_post else (st,)
        k, _ = metropolis_zufallsweg(ld, start, schritt, laenge, rng)
        ketten.append(k[:, 0])
    C = np.array(ketten)
    zeile(name, round(r_dach(C[:, laenge // 2:]), 3), breite=34)      # zweite Hälfte jeder Kette
buch("1,003 / 1,001 / 6,9 / 1,006 -- Faustregel: unter 1,01")

kopf("Das wirksame n der Kette (8.3), 20 000 Werte")
zeile("Verfahren", "tau", "n_eff", "SE von mu", breite=16)
g = gibbs_normal(y, 20_000, rng)[:, 0]
for name, kette in (("Gibbs", g),
                    ("Metropolis 0,1", metropolis_zufallsweg(log_post, (ybar, math.log(s)), 0.1, 20_000, rng)[0][:, 0]),
                    ("Metropolis 0,01", metropolis_zufallsweg(log_post, (ybar, math.log(s)), 0.01, 20_000, rng)[0][:, 0])):
    tau = tau_int(kette, max_lag=2000)
    zeile(name, round(tau, 1), round(len(kette) / tau, 0), round(kette.std() / math.sqrt(len(kette) / tau), 4), breite=16)
print(f"Planung: Standardfehler 0,001 bei Streuung 0,053 und tau 6,4 braucht (0,053/0,001)^2 * 6,4 = {(0.053 / 0.001) ** 2 * 6.4:,.0f} Schritte")
buch("Gibbs 1,0 / 20 000 / 0,0004; Metropolis 0,1: 6,4 / 2 800 / 0,0010; 0,01: 108 / 170 / 0,0041")

kopf("Der Bruchfall: vier Ketten, ein Gipfel (8.5)")
C = np.array([metropolis_zufallsweg(log_bimodal, (st,), 1.0, 5000, rng)[0][:, 0] for st in (5, 5.5, 6, 6.5)])
print(f"Starts 5 / 5,5 / 6 / 6,5: R-Dach = {r_dach(C[:, 2500:]):.3f}, Mittel {np.round(C.mean(axis=1), 2)}, wahres Mittel 0")
buch("R-Dach 1,00 -- jede Diagnostik gibt grünes Licht")

kopf("Aufgabe 8.1 -- R-Dach von Hand")
m = np.array([1.279, 1.286, 1.278, 1.290]); v = np.array([0.00281, 0.00279, 0.00285, 0.00276]); nn = 1000
W = v.mean(); B = nn * m.var(ddof=1); V = (nn - 1) / nn * W + B / nn
print(f"W = {W:.6f}, B = {B:.4f}, V = {V:.6f}, R-Dach = {math.sqrt(V / W):.3f}")
B2 = nn * 0.05 ** 2; V2 = (nn - 1) / nn * W + B2 / nn
print(f"Gegenprobe mit Streuung der Mittel 0,05: B = {B2:.2f}, R-Dach = {math.sqrt(V2 / W):.2f}")
buch("1,005; Gegenprobe 1,38")

kopf("Aufgabe 8.2 -- vier Kerne, zehn Minuten, 10 000 Schritte je Minute, tau 20, sigma 2")
print(f"400 000 Schritte, erste Hälfte verworfen: n_eff = 200 000 / 20 = 10 000, SE = 2 / sqrt(10 000) = {2 / math.sqrt(10_000):.3f}")
buch("0,02 -- eine Kette je Kern")
