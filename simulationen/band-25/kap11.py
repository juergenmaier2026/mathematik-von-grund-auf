"""Band 25, Kapitel 11 -- Das Itô-Integral.

Erzeugt: int W dW ausgewertet am linken Rand (Itô) und in der Mitte (Stratonowitsch)
gegen die Formeln (W_T^2 - T)/2 und W_T^2/2 (11.2).
"""
import _pfad
import numpy as np
from mvg.prozesse import brownsche_pfade
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2511)
W = brownsche_pfade(rng, 1, 100_000)[0]; dW = np.diff(W); T = 1.0

kopf("int_0^1 W dW an einem Pfad mit 100 000 Schritten (11.2)")
ito = (W[:-1] * dW).sum()
strat = ((W[:-1] + W[1:]) / 2 * dW).sum()
zeile("Auswertung", "Formel", "simuliert", breite=16)
zeile("links (Itô)", round((W[-1] ** 2 - T) / 2, 3), round(float(ito), 3), breite=16)
zeile("Mitte (Strat.)", round(W[-1] ** 2 / 2, 3), round(float(strat), 3), breite=16)
zeile("Differenz", round(T / 2, 3), round(float(strat - ito), 3), breite=16)
buch("-0,207 / 0,292 / 0,499 -- der Auswertungspunkt entscheidet; die Differenz ist T/2")
