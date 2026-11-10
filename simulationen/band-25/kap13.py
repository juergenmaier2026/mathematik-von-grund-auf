"""Band 25, Kapitel 13 -- Optionen.

Erzeugt: die Absicherung einer Kaufoption in endlich vielen Umschichtungen (13.4):
Streuung des Absicherungsfehlers gegen die Zahl der Umschichtungen, 20 000 Pfade.
"""
import _pfad
import math
import numpy as np
from scipy.stats import norm
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2513)
S0 = K = 100.0; r = 0.02; sg = 0.20; T = 1.0

def preis(S, t):
    tau = T - t
    if tau <= 0: return np.maximum(S - K, 0), (S > K).astype(float)
    d1 = (np.log(S / K) + (r + sg ** 2 / 2) * tau) / (sg * math.sqrt(tau)); d2 = d1 - sg * math.sqrt(tau)
    return S * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2), norm.cdf(d1)

C0, _ = preis(np.array([S0]), 0.0)
print(f"Black-Scholes-Preis: {C0[0]:.3f}")
kopf("Absicherungsfehler bei n Umschichtungen, 20 000 Pfade (13.4)")
zeile("Umschichtungen", "Streuung", "Anteil am Preis", "mittlerer Fehler", breite=18)
for n in (4, 20, 100, 500):
    dt = T / n; S = np.full(20_000, S0); kasse = np.full(20_000, C0[0]); delta = preis(S, 0.0)[1]
    kasse -= delta * S                                            # Delta Aktien kaufen, Rest in der Kasse
    for k in range(1, n + 1):
        S = S * np.exp((r - sg ** 2 / 2) * dt + sg * math.sqrt(dt) * rng.standard_normal(20_000))
        kasse *= math.exp(r * dt)
        neu = preis(S, k * dt)[1] if k < n else np.zeros_like(S)
        kasse -= (neu - delta) * S; delta = neu                   # umschichten
    fehler = kasse - np.maximum(S - K, 0)                         # Endkasse gegen Verpflichtung
    zeile(n, round(float(fehler.std()), 2), f"{fehler.std() / C0[0]:.0%}", round(float(fehler.mean()), 3), breite=18)
buch("3,23 (36 %) / 1,53 (17 %) / 0,70 (8 %) / 0,32 (4 %); mittlerer Fehler praktisch null")
