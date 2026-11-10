"""Band 26, Kapitel 14 -- Der Leitfall, alle Verfahren.

Erzeugt: die Tabelle der Verfahren auf dem Leitfall -- schlicht, mit Kontrolle,
Experiment, Instrument, feste Effekte, Ereignisstudie -- mit denselben Daten
wie in den Kapiteln 1, 3 und 7.
"""
import _pfad
import numpy as np
from mvg.versicherte import querschnitt, experiment, reform, panel
from mvg.oekonometrie import kq, steigung, zweistufig, innerhalb
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2614)
kopf("Sechs Verfahren, ein wahrer Wert -0,100 (14.1)")
g, x, y = querschnitt(rng, 200_000)
zeile("schlichte Regression", round(steigung(y, x), 3), breite=34)
zeile("Kontrolle für die Gesundheit", round(kq(y, x, g)[0][1], 3), breite=34)
g2, x2, y2 = experiment(rng, 200_000)
zeile("Experiment", round(steigung(y2, x2), 3), breite=34)
g3, x3, y3, z3 = reform(rng, 200_000)
zeile("Reform als Instrument", round(zweistufig(y3, x3, z3)[0], 3), breite=34)
xp, yp = panel(rng, 5000)
n, T = xp.shape
D = np.repeat(np.eye(T)[:, 1:][None], n, axis=0).reshape(n * T, T - 1)
zeile("feste Effekte mit Jahreseffekten", round(kq(innerhalb(yp).ravel(), innerhalb(xp).ravel(), D)[0][1], 3), breite=34)
buch("alle Verfahren treffen, weil in der Simulation alle Annahmen gelten -- in Wirklichkeit gilt je eine")
