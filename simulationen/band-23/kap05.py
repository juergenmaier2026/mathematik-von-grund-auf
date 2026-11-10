"""Band 23, Kapitel 5 -- Anpassen.

Erzeugt: die Verstärkung der Verzerrung durch Anpassung an ein Instrument
(Aufgabe 5.3): X = 1,2 Z + 0,5 U + e_X, Y = X + U + e_Y, U unbeobachtet.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import kq, steigung
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2305)
n = 1_000_000
Z, U = rng.standard_normal(n), rng.standard_normal(n)
X = 1.2 * Z + 0.5 * U + rng.standard_normal(n)
Y = 1.0 * X + 1.0 * U + rng.standard_normal(n)
kopf("Verzerrung der Regression von Y auf X, wahrer Effekt 1,0 (Aufgabe 5.3)")
zeile("", "Formel", "simuliert", breite=16)
zeile("ohne Z", round(0.5 / 2.69, 3), round(steigung(Y, X) - 1, 3), breite=16)
zeile("mit Z", round(0.5 / 1.25, 3), round(kq(Y, X, Z)[0][1] - 1, 3), breite=16)
buch("0,186 / 0,186 und 0,400 / 0,403 -- das Instrument erklärt den guten Teil der Variation von X und lässt den schlechten übrig")
