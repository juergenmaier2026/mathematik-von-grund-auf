"""Band 26, Kapitel 11 -- Scheinregression.

Erzeugt: Ablehnungsraten bei zwei unabhängigen Irrfahrten (11.1), den
Dickey-Fuller-Wert aus 10 000 Irrfahrten (Satz 11.2) und den Bruchfall mit
trägen stationären Reihen (11.4).
"""
import _pfad
import numpy as np
from mvg.oekonometrie import ar1, kq, dickey_fuller_t
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2611)

def ablehnungsrate(phi, T, R=2000):
    treffer = 0
    for _ in range(R):
        x, y = ar1(rng, phi, T), ar1(rng, phi, T)
        b, res, se = kq(y, x)
        treffer += abs(b[1] / se[1]) > 1.96
    return treffer / R

kopf("Zwei unabhängige Irrfahrten: Anteil 'signifikanter' Zusammenhänge bei 5 % Niveau (11.1)")
zeile("T", "Ablehnungsrate", breite=16)
for T in (50, 100, 500):
    zeile(T, f"{ablehnungsrate(1.0, T):.1%}", breite=16)
buch("67,7 % / 76,1 % / 88,9 % -- mehr Daten, mehr falsche Befunde")

kopf("Satz 11.2 -- Dickey-Fuller: t-Statistik von gamma unter phi = 1, 10 000 Irrfahrten der Länge 100")
t = np.array([dickey_fuller_t(ar1(rng, 1.0, 100)) for _ in range(10_000)])
print(f"5-%-Quantil {np.percentile(t, 5):.2f} (normal: -1,64), Mittel {t.mean():.2f} (normal: 0)")
buch("-2,90 und -1,53")

kopf("Der Bruchfall: auch stationäre, träge Reihen täuschen (11.4), T = 100")
zeile("Reihen", "Ablehnungsrate", breite=20)
zeile("zwei Irrfahrten", f"{ablehnungsrate(1.0, 100):.1%}", breite=20)
zeile("zwei AR(0,9)", f"{ablehnungsrate(0.9, 100):.1%}", breite=20)
zeile("zwei weiße Rauschen", f"{ablehnungsrate(0.0, 100):.1%}", breite=20)
buch("76,9 % / 50,3 % / 5 %")
