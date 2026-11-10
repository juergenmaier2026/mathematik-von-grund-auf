"""Band 26, Kapitel 4 -- Momente.

Erzeugt: zwei Momentenschätzer für lambda bei 5000 poissonverteilten Besuchszahlen (Aufgabe 4.1).
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2604)
y = rng.poisson(3.2, 5000)
kopf("Aufgabe 4.1 -- Momentenschätzer aus Mittelwert und Varianz, lambda = 3,2")
zeile("Moment", "lambda-Dach", breite=14)
zeile("Mittelwert", round(float(y.mean()), 3), breite=14)
zeile("Varianz", round(float(y.var(ddof=1)), 3), breite=14)
buch("3,234 und 3,290 -- zwei Momente, zwei Schätzer, und GMM gewichtet sie")
