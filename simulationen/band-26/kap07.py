"""Band 26, Kapitel 7 -- Jeder sein eigener Vergleich.

Erzeugt: die vier Antworten -- gepoolt, zwischen, innerhalb, mit Jahreseffekten --
für 5000 Versicherte über sechs Jahre (7.1) und den Bruchfall mit Messfehler (7.4).
"""
import _pfad
import numpy as np
from mvg.versicherte import panel
from mvg.oekonometrie import kq, steigung, innerhalb
from mvg.ausgabe import kopf, zeile, buch

def vier_antworten(x, y):
    n, T = x.shape
    gepoolt = steigung(y.ravel(), x.ravel())
    zwischen = steigung(y.mean(axis=1), x.mean(axis=1))
    xi, yi = innerhalb(x), innerhalb(y)
    innen = steigung(yi.ravel(), xi.ravel())
    D = np.repeat(np.eye(T)[:, 1:][None], n, axis=0).reshape(n * T, T - 1)       # Jahresdummies
    zweiseitig = kq(yi.ravel(), xi.ravel(), D)[0][1]
    return gepoolt, zwischen, innen, zweiseitig

rng = np.random.default_rng(2607)
kopf("5000 Versicherte, sechs Jahre: vier Antworten (7.1)")
x, y = panel(rng, 5000)
for name, wert in zip(("gepoolt", "zwischen (Personenmittel)", "innerhalb (Abweichungen)", "innerhalb, mit Jahreseffekten"), vier_antworten(x, y)):
    zeile(name, round(wert, 3), breite=30)
zeile("wahrer Wert", -0.1, breite=30)
buch("-0,302 / -0,418 / -0,070 / -0,099")

kopf("Der Bruchfall: Messfehler in der Zuzahlung (Streuung 3 Euro) wird innerhalb verstärkt (7.4)")
rng = np.random.default_rng(2607)
xm, ym = panel(rng, 5000, messfehler=3.0)
o = vier_antworten(x, y); m = vier_antworten(xm, ym)
zeile("Verfahren", "ohne Fehler", "mit Fehler", breite=16)
zeile("gepoolt", round(o[0], 3), round(m[0], 3), breite=16)
zeile("innerhalb + Jahre", round(o[3], 3), round(m[3], 3), breite=16)
buch("-0,302 / -0,282 und -0,099 / -0,059 -- die Innerhalb-Varianz von x ist kleiner, der Fehler bleibt")
