"""Band 27, Kapitel 12 -- Was jetzt gerade los ist.

Erzeugt: den Teilchenfilter für die schwankende Ankunftsrate (12.2), die
Degeneration ohne Neuziehung (12.3), die Überdeckung der Intervalle und den
Bruchfall mit falschem Modell (12.5) sowie den Filterschritt mit vier
Teilchen (Aufgabe 12.1).
"""
import _pfad
import math
import numpy as np
from scipy.special import gammaln
from mvg.filter import teilchenfilter, kalman_1d
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(np.random.PCG64(12))          # derselbe Startwert wie im Buch
T, s = 200, 0.05
loglam = math.log(2.4) + np.cumsum(rng.normal(0, s, T))   # verborgener Zustand
lam = np.exp(loglam)
y = rng.poisson(lam)                                      # tägliche Zählungen
print(f"lambda_t zwischen {lam.min():.2f} und {lam.max():.2f}; Zählungen im Mittel {y.mean():.2f}")

log_lik = lambda yt, x: yt * x - np.exp(x) - gammaln(yt + 1)
schritt = lambda x, r: x + r.normal(0, s, len(x))
start = lambda r, N: math.log(2.4) + r.normal(0, 0.3, N)
rmse = lambda a: math.sqrt(np.mean((a[20:] - lam[20:]) ** 2))

kopf("Vier Schätzer für lambda_t über 200 Tage (12.2)")
ma = np.array([y[max(0, t - 6):t + 1].mean() for t in range(T)])
kf = kalman_1d(np.sqrt(y + 3 / 8), q=0.04 ** 2, r=0.25, m0=math.sqrt(2.4), P0=0.1) ** 2 - 3 / 8
def filter_lambda(N, seed, s_modell=s, neu=True):
    f, lo, hi, ne = teilchenfilter(y, log_lik, lambda x, r: x + r.normal(0, s_modell, len(x)), start, N, np.random.default_rng(seed), neu)
    return np.exp(f), np.exp(lo), np.exp(hi), ne        # der Filter läuft auf log lambda; exp des Mittels ist das geometrische Mittel
pf, lo, hi, neff = filter_lambda(1000, 1)
zeile("Schätzer", "Wurzel des mittleren Fehlerquadrats", breite=36)
zeile("die Zählung Y_t selbst", round(rmse(y.astype(float)), 2), breite=36)
zeile("Sieben-Tage-Mittel", round(rmse(ma), 2), breite=36)
zeile("Kalman-Filter auf sqrt(Y + 3/8)", round(rmse(kf), 2), breite=36)
zeile("Teilchenfilter (1000 Teilchen)", round(rmse(pf), 2), breite=36)
ueberdeckung = np.mean((lo[20:] <= lam[20:]) & (lam[20:] <= hi[20:]))
print(f"90-%-Intervall: mittlere Breite {np.mean(hi[20:] - lo[20:]):.2f}, Überdeckung {ueberdeckung:.0%}")
buch("1,59 / 0,60 / 0,45 / 0,42; Breite 1,5, Überdeckung 94 %")

kopf("Die Degeneration ohne Neuziehung (12.3)")
_, _, _, ne_ohne = filter_lambda(1000, 1, neu=False)
zeile("Tag", 1, 5, 20, 50, 200, breite=8)
zeile("n_eff", *[round(ne_ohne[i], 0) for i in (0, 4, 19, 49, 199)], breite=8)
print(f"mit Neuziehung: n_eff im Mittel {neff.mean():.0f}, nie unter {neff.min():.0f}")
pf_ohne = filter_lambda(1000, 1, neu=False)[0]
print(f"Fehler ohne Neuziehung {rmse(pf_ohne):.2f} statt {rmse(pf):.2f}")
for N in (50, 200, 5000):
    print(f"   N = {N:5d}: Fehler {rmse(filter_lambda(N, 3)[0]):.2f}")
buch("712 / 389 / 196 / 39 / 2; mit Neuziehung im Mittel 929; N = 50 reicht")

kopf("Der Bruchfall: das falsche Modell filtert trotzdem (12.5)")
zeile("angenommenes s", "Fehler", "Überdeckung 90 %", breite=18)
for s_m in (0.01, 0.05, 0.20):
    f, lo, hi, _ = filter_lambda(2000, 5, s_modell=s_m)
    u = np.mean((lo[20:] <= lam[20:]) & (lam[20:] <= hi[20:]))
    zeile(s_m, round(rmse(f), 2), f"{u:.0%}", breite=18)
buch("0,41 / 66 %; 0,42 / 94 %; 0,64 / 100 %")

kopf("Aufgabe 12.1 -- ein Filterschritt mit vier Teilchen, Y_t = 4")
l = np.array([1.5, 2.0, 3.0, 4.0]); v = l ** 4 * np.exp(-l); w = v / v.sum()
print(f"Likelihood lambda^4 e^-lambda: {np.round(v, 2)}, Gewichte {np.round(w, 3)}")
print(f"Filtermittel {w @ l:.2f} (vorher {l.mean():.3f}); n_eff = {1 / np.sum(w * w):.1f} von 4")
buch("Gewichte 0,094 / 0,181 / 0,336 / 0,390; Mittel 3,07; n_eff 3,3")
