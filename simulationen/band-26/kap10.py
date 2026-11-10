"""Band 26, Kapitel 10 -- Zeitreihen.

Erzeugt: Varianz und Autokorrelation des AR(1) für phi = 0,5 / 0,9 / 1,0 (10.2)
und die Hurwicz-Verzerrung von phi-Dach für T = 20, 50, 200 (10.3), 5000 Wiederholungen.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import ar1, steigung
from mvg.statistik import autokorrelation
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2610)

kopf("AR(1) über 200 Beobachtungen: Varianz und Autokorrelation (10.2)")
zeile("phi", "Var theor.", "Var Stichpr.", "rho_1", "rho_5", breite=14)
for phi in (0.5, 0.9, 1.0):
    y = ar1(rng, phi, 1200)[1000:]
    r = autokorrelation(y, 5)
    zeile(phi, "unendlich" if phi == 1 else round(1 / (1 - phi ** 2), 2), round(float(y.var()), 2), round(r[0], 2), round(r[4], 2), breite=14)
buch("1,33 / 1,37 / 0,57 / 0,10;  5,26 / 3,72 / 0,87 / 0,51;  unendlich / 9,62 / 0,94 / 0,77 (eine Reihe -- die Werte streuen stark)")

kopf("Hurwicz: phi-Dach bei phi = 0,9, 5000 Wiederholungen (10.3)")
zeile("T", "Mittel phi-Dach", "Näherung phi-(1+3phi)/T", "wahr", breite=24)
for T in (20, 50, 200):
    sch = [steigung(y[1:], y[:-1]) for y in (ar1(rng, 0.9, T + 200)[200:] for _ in range(5000))]
    zeile(T, round(float(np.mean(sch)), 3), round(0.9 - (1 + 2.7) / T, 3), 0.9, breite=24)
buch("0,674 / 0,808 / 0,878 gegen 0,715 / 0,826 / 0,882")
