"""Band 27, Kapitel 9 -- Fliegen statt kriechen, und Pfade rechnen.

Erzeugt: Zufallsweg gegen Hamilton in hundert Dimensionen (9.1, 9.2),
Euler-Maruyama und Milstein für die geometrische Brownsche Bewegung (Satz 9.4),
die Mehrebenen-Tabelle (Satz 9.5), die Stabilitätsklippe (9.5) und den
Sprungschritt von Hand (Aufgabe 9.1).
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm
from mvg.mcmc import metropolis_zufallsweg, hamilton
from mvg.sde import euler_maruyama, milstein, brownsche_zuwaechse, vergroebern, mehrebenen_pfadzahlen
from mvg.statistik import tau_int
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2709)

# Ziel: hundert unabhängige Normalverteilungen mit Streuungen 0,1 bis 1,0
d = 100
sig = np.linspace(0.1, 1.0, d)
U = lambda x: 0.5 * np.sum((x / sig) ** 2)
grad_U = lambda x: x / sig ** 2
log_dichte = lambda x: -U(x)

kopf("Der Zufallsweg in hundert Dimensionen (9.1): 100 000 Schritte, 20 000 verworfen")
zeile("Schritt", "Annahme", "tau (sigma=0,1)", "tau (sigma=1)", breite=16)
for f in (0.1, 0.2, 0.3):
    schritt = 2.38 / math.sqrt(d) * f
    k, acc = metropolis_zufallsweg(log_dichte, np.zeros(d), schritt, 100_000, rng)
    k = k[20_000:]
    zeile(round(schritt, 3), f"{acc:.0%}", round(tau_int(k[:, 0], max_lag=5000), 0), round(tau_int(k[:, -1], max_lag=5000), 0), breite=16)
buch("Schritt 0,071: Annahme 26 %, tau 34 und 2 000 -- 40 wirksame Werte aus 80 000")

kopf("Hamilton-Monte-Carlo (Satz 9.2): 5 000 Schritte, L = 20, eps = 0,1")
k, acc, dH = hamilton(U, grad_U, np.zeros(d), 0.1, 20, 5000, rng)
print(f"Annahme {acc:.0%}, tau (sigma=0,1) {tau_int(k[:, 0], max_lag=500):.1f}, tau (sigma=1) {tau_int(k[:, -1], max_lag=500):.1f},"
      f" Streuungen {k[:, 0].std():.3f} und {k[:, -1].std():.3f}; 100 000 Gradienten")
buch("Annahme 85 %, tau 1,0 -- das Hundertfache an wirksamen Werten bei gleichem Aufwand")

kopf("Die Stabilitätsklippe bei eps = 2 sigma_min = 0,2 (Bruchfall 9.5)")
x0 = rng.standard_normal(d) * sig
zeile("eps", "Annahme", "max Delta H", breite=14)
for eps in (0.10, 0.15, 0.19, 0.21):
    _, acc, dH = hamilton(U, grad_U, x0, eps, 20, 1000, rng)
    zeile(eps, f"{acc:.0%}", f"{dH.max():.1e}", breite=14)
buch("85 % / 61 % / 43 % / 0 % -- bei 0,21 Delta H bis 10^12")

kopf("Satz 9.4 -- geometrische Brownsche Bewegung, X0 = 100, a = 0,05 X, b = 0,2 X, T = 1, 20 000 Pfade")
X0, mu_, sg = 100.0, 0.05, 0.2
a = lambda x: mu_ * x; b = lambda x: sg * x; db = lambda x: sg * np.ones_like(x)
dW = brownsche_zuwaechse(rng, 20_000, 1024, 1.0)
exakt = X0 * np.exp((mu_ - sg * sg / 2) + sg * dW.sum(axis=1))      # derselbe Brownsche Pfad
zeile("Schritte", "stark Euler", "stark Milstein", "schwach Euler (exakt)", breite=22)
for n in (4, 16, 64, 256, 1024):
    dWn = vergroebern(dW, n)
    e = euler_maruyama(a, b, X0, 1.0, n, dWn)
    m = milstein(a, b, db, X0, 1.0, n, dWn)
    schwach = X0 * math.exp(mu_) - X0 * (1 + mu_ / n) ** n                 # Zinseszins gegen stetigen Zins
    zeile(n, round(float(np.sqrt(np.mean((e - exakt) ** 2))), 4), round(float(np.sqrt(np.mean((m - exakt) ** 2))), 4), round(schwach, 5), breite=22)
buch("Euler halbiert je Vervierfachung, Milstein und der schwache Fehler vierteln")

kopf("Satz 9.5 -- Mehrebenen-Monte-Carlo für die Kaufoption max(X_T - 100, 0), abgezinst")
K, disc = 100.0, math.exp(-mu_)
d1 = (math.log(X0 / K) + (mu_ + sg * sg / 2)) / sg; d2 = d1 - sg
print(f"Black-Scholes exakt: {X0 * norm.cdf(d1) - K * disc * norm.cdf(d2):.3f}")
dW = brownsche_zuwaechse(rng, 20_000, 64, 1.0)
payoff = lambda x: disc * np.maximum(x - K, 0)
V, C, vorher = [], [], None
zeile("Ebene", "Schritte", "Var[P_l]", "Var[P_l - P_l-1]", "Mittel der Differenz", breite=20)
for l in range(7):
    P = payoff(euler_maruyama(a, b, X0, 1.0, 2 ** l, vergroebern(dW, 2 ** l)))
    if vorher is None:
        V.append(P.var()); zeile(l, 2 ** l, round(P.var(), 1), "", round(P.mean(), 3), breite=20)
    else:
        D = P - vorher; V.append(D.var()); zeile(l, 2 ** l, round(P.var(), 1), round(D.var(), 4), round(D.mean(), 4), breite=20)
    C.append(2 ** l); vorher = P
N = mehrebenen_pfadzahlen(V, C, 0.02)
print(f"Pfadzahlen für eps = 0,02: {N.tolist()}")
print(f"Kosten Mehrebenen {np.sum(N * np.array(C)):,.0f} Schritte; eine Ebene mit 64 Schritten: {int(vorher.var() / (0.02 ** 2 / 2)) * 64:,.0f} Schritte")
buch("Faktor 19")

kopf("Aufgabe 9.1 -- ein Sprungschritt von Hand: U = x^2/2, x = 1, p = 0, eps = 0,5")
x, p, eps = 1.0, 0.0, 0.5
p -= 0.5 * eps * x; x += eps * p; p -= 0.5 * eps * x
print(f"x' = {x:.3f}, p' = {p:.3f}, H0 = 0,5, H1 = {0.5 * x * x + 0.5 * p * p:.3f}; exakt x = cos 0,5 = {math.cos(0.5):.3f}, p = -sin 0,5 = {-math.sin(0.5):.3f}")
buch("0,875 / -0,469 / Delta H = -0,007")
