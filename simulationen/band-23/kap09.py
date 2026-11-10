"""Band 23, Kapitel 9 -- Instrumente.

Erzeugt: starke, mäßige und schwache Instrumente an 400 Wiederholungen mit
n = 500, wahrer Effekt 1,000 (9.4): Median und Streuung des Quotientenschätzers.
Modell: X = a Z + 0,5 U + e_X, Y = X + U + e_Y; a = 1,2 / 0,25 / 0,06.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import zweistufig
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2309)
kopf("Schwache Instrumente, 400 Wiederholungen, n = 500 (9.4)")
zeile("Stärke", "a", "F (Median)", "Median", "Streuung", breite=12)
for name, a in (("stark", 1.2), ("mäßig", 0.25), ("schwach", 0.06)):
    sch, Fs = [], []
    for _ in range(400):
        Z, U = rng.standard_normal(500), rng.standard_normal(500)
        X = a * Z + 0.5 * U + rng.standard_normal(500); Y = X + U + rng.standard_normal(500)
        b, F = zweistufig(Y, X, Z); sch.append(b); Fs.append(F)
    sch = np.array(sch)
    zeile(name, a, round(float(np.median(Fs)), 0), round(float(np.median(sch)), 3), round(float(sch.std()), 2), breite=12)
buch("F 398 / 16 / 1; Median 1,010 / 0,997 / 1,172; Streuung 0,07 / 0,34 / 16,8")
