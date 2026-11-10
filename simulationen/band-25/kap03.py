"""Band 25, Kapitel 3 -- Stationäre Verteilungen.

Erzeugt: die Einschwingphase einer Kette mit |lambda_2| nahe 1 (3.4) und das
Verfahren aus Aufgabe 3.3 (Metropolis auf endlich vielen Zuständen).
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2503)

kopf("Einschwingen: Zwei-Zustands-Kette mit lambda_2 = 0,99 (3.4)")
P = np.array([[0.995, 0.005], [0.005, 0.995]])             # lambda_2 = 0,99
pi = np.array([0.5, 0.5]); v = np.array([1.0, 0.0])
zeile("Schritte", "P(Zustand 1)", "Abstand zu pi", breite=16)
for n in (10, 100, 300, 500, 1000):
    w = v @ np.linalg.matrix_power(P, n)
    zeile(n, round(w[0], 4), f"{abs(w[0] - 0.5):.1e}", breite=16)
buch("mehrere hundert Schritte, bis die Einschwingphase vorbei ist; Abstand fällt wie 0,99^n")

kopf("Aufgabe 3.3 -- eine Verteilung erzeugen: pi = (0,1, 0,2, 0,3, 0,4) auf vier Zuständen")
pi = np.array([0.1, 0.2, 0.3, 0.4])
x, zaehl = 0, np.zeros(4)
for _ in range(100_000):
    y = (x + rng.choice([-1, 1])) % 4                        # symmetrischer Vorschlag: Nachbar auf dem Kreis
    if rng.random() < min(1, pi[y] / pi[x]): x = y
    zaehl[x] += 1
zeile("Zustand", 0, 1, 2, 3, breite=10)
zeile("gewünscht", *pi, breite=10)
zeile("Kette", *np.round(zaehl / zaehl.sum(), 3), breite=10)
buch("die Kette trifft pi -- ohne die Normierungskonstante je zu benutzen")
