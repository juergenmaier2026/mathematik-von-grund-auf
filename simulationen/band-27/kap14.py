"""Band 27, Kapitel 14 -- Was man dem Ergebnis glauben darf.

Erzeugt: die A-posteriori-Vorhersage der Abweisungsrate (Definition 14.1),
die Sobol-Indizes (Satz 14.3), die Entscheidungstabelle (14.4), die
Elastizität (Aufgabe 14.1) und den Wert von vierhundert Tagen (Aufgabe 14.3).
"""
import _pfad
import math
import numpy as np
from mvg.erlang import erlang_b
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(np.random.PCG64(14))
N = 20_000
# Daten: 100 Aufenthalte mit ybar = ln 4, s = 0,6 (logarithmische Skala); 240 Ankünfte in 100 Tagen
n, ybar, s2, k, D = 100, math.log(4), 0.6 ** 2, 240, 100

def ziehungen(n, k, D):
    """A-posteriori-Ziehungen von (lambda, mu, sigma) bei flachen A-priori-Dichten (Band 21)."""
    sigma = np.sqrt((n - 1) * s2 / rng.chisquare(n - 1, N))
    mu = rng.normal(ybar, sigma / math.sqrt(n))
    lam = rng.gamma(k + 1, 1 / D, N)
    return lam, mu, sigma

lam, mu, sigma = ziehungen(n, k, D)
a = lam * np.exp(mu + sigma ** 2 / 2)
p12 = np.array([erlang_b(x, 12) for x in a]); p13 = np.array([erlang_b(x, 13) for x in a])

kopf("Definition 14.1 -- die Abweisungsrate unter Parameterunsicherheit, 20 000 Ziehungen")
zeile("", "Mittel", "Streuung", "5 %", "95 %", "P(g > 0,2)", breite=12)
for name, p in (("12 Betten", p12), ("13 Betten", p13), ("Differenz", p12 - p13)):
    zeile(name, round(p.mean(), 4), round(p.std(), 4), round(np.percentile(p, 5), 4), round(np.percentile(p, 95), 4), round(np.mean(p > 0.2), 3), breite=12)
buch("0,183 +- 0,042 [0,117; 0,254] 0,33 / 0,141 +- 0,038 / Differenz 0,042 +- 0,0035")

kopf("Satz 14.3 -- Sobol-Haupteffekte für die Rate bei 12 Betten (50 Klassen je Parameter)")
def S1(x, p, klassen=50):
    q = np.quantile(x, np.linspace(0, 1, klassen + 1))
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, klassen - 1)
    cm = np.array([p[idx == b].mean() for b in range(klassen)])
    return cm.var() / p.var()
for name, x in (("lambda", lam), ("mu", mu), ("sigma", sigma)):
    print(f"   S({name:6s}) = {S1(x, p12):.2f}")
buch("0,48 / 0,43 / 0,09 -- Summe 1: keine Wechselwirkungen, weil ln a eine Summe ist")

kopf("Die Entscheidungstabelle (14.4)")
print(f"Liegt die Rate über 20 %?                         {np.mean(p12 > 0.2):.0%}")
print(f"Bringt das 13. Bett sie unter 15 %?               {np.mean(p13 < 0.15):.0%}")
print(f"Senkt es die Rate um mehr als drei Punkte?        {np.mean(p12 - p13 > 0.03):.1%}")
buch("33 % / 62 % / > 99 %")

kopf("Aufgabe 14.1 -- Elastizität und Deltamethode")
a0 = 2.4 * math.exp(ybar + s2 / 2)
dB = (erlang_b(a0 + 0.01, 12) - erlang_b(a0 - 0.01, 12)) / 0.02
el = a0 / erlang_b(a0, 12) * dB
rel_a = math.sqrt((lam.std() / lam.mean()) ** 2 + (mu.std()) ** 2 + (sigma ** 2 / 2).var())
print(f"dB/da = {dB:.3f}, Elastizität {el:.2f}; relative Streuung von a {rel_a:.3f}; Deltamethode {el * rel_a * erlang_b(a0, 12):.4f} gegen simuliert {p12.std():.4f}")
for rho_ in (0.8, 0.6):
    aa = 12 * rho_
    print(f"   bei Auslastung {rho_}: Elastizität {aa / erlang_b(aa, 12) * (erlang_b(aa + .01, 12) - erlang_b(aa - .01, 12)) / 0.02:.1f}")
buch("2,6; 0,092; 0,043 gegen 0,042; bei 80 % 3,4, bei 60 % 5")

kopf("Aufgabe 14.3 -- vierhundert Tage Daten")
lam4, mu4, sigma4 = ziehungen(400, 960, 400)
p4 = np.array([erlang_b(x, 12) for x in lam4 * np.exp(mu4 + sigma4 ** 2 / 2)])
print(f"Streuung {p4.std():.4f} statt {p12.std():.4f}; 90-%-Intervall [{np.percentile(p4, 5):.3f}; {np.percentile(p4, 95):.3f}]")
buch("0,021 -- die Hälfte")
