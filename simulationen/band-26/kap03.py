"""Band 26, Kapitel 3 -- Instrumente.

Erzeugt: die Reform als Instrument bei 200 000 Versicherten (3.2) und den Bruchfall
der schwachen Instrumente (2000 Versicherte, 500 Wiederholungen).
"""
import _pfad
import numpy as np
from mvg.versicherte import reform
from mvg.oekonometrie import steigung, zweistufig
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2603)

kopf("Die Reform als Instrument: 200 000 Versicherte, Erhöhung um 7,83 Euro (3.2)")
g, x, y, z = reform(rng, 200_000)
iv, F = zweistufig(y, x, z)
zeile("Verfahren", "je Euro", breite=26)
zeile("kleinste Quadrate", round(steigung(y, x), 3), breite=26)
zeile("zweistufig", round(iv, 3), breite=26)
zeile("Quotient (Satz 3.1)", round((y[z == 1].mean() - y[z == 0].mean()) / (x[z == 1].mean() - x[z == 0].mean()), 3), breite=26)
zeile("wahrer Wert", -0.1, breite=26)
buch("-0,297 / -0,100 / -0,100")

kopf("Der Bruchfall: 2000 Versicherte, 500 Wiederholungen, schwächer werdende Reform (3.5)")
zeile("Wirkung", "Erhöhung", "F", "Median", "5 %", "95 %", breite=10)
for name, delta in (("stark", 7.83), ("mäßig", 2.0), ("schwach", 0.5)):
    sch, Fs = [], []
    for _ in range(500):
        g, x, y, z = reform(rng, 2000, delta)
        b, F = zweistufig(y, x, z); sch.append(b); Fs.append(F)
    sch = np.array(sch)
    zeile(name, delta, round(float(np.median(Fs)), 1), round(float(np.median(sch)), 3), round(float(np.percentile(sch, 5)), 3), round(float(np.percentile(sch, 95)), 3), breite=10)
buch("F 719 / 44 / 2,8; Median -0,098 / -0,100 / -0,135; 5-95 % [-0,121; -0,073] / [-0,177; +0,019] / [-0,805; +0,922]")
