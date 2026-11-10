"""Band 26, Kapitel 1 -- Wenn der Regressor nicht unschuldig ist.

Erzeugt: die schlichte Regression gegen die wahre Wirkung (1.1), die Kontrolle
für die Gesundheit (1.2), das Experiment (1.3) und die Verzerrungsformel (Aufgabe 1.1).
"""
import _pfad
import numpy as np
from mvg.versicherte import querschnitt, experiment
from mvg.oekonometrie import kq, steigung
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2601)
g, x, y = querschnitt(rng, 200_000)

kopf("Der Leitfall: 200 000 Versicherte (1.1, 1.2)")
zeile("Modell", "je Euro Zuzahlung", breite=26)
zeile("nur Zuzahlung", round(steigung(y, x), 3), breite=26)
zeile("Zuzahlung und Gesundheit", round(kq(y, x, g)[0][1], 3), breite=26)
zeile("wahrer Wert", -0.1, breite=26)
buch("-0,376 / -0,098 / -0,100")

kopf("Das Experiment: zufällige Zuweisung von 0, 10, 20, 30 Euro (1.3)")
g2, x2, y2 = experiment(rng, 200_000)
zeile("Zuzahlung", "Besuche je Jahr", breite=18)
for s in (0, 10, 20, 30):
    zeile(f"{s} Euro", round(float(y2[x2 == s].mean()), 2), breite=18)
print(f"Steigung {steigung(y2, x2):.4f} gegen den wahren Wert -0,100")
buch("8,01 / 7,00 / 5,99 / 5,02; Steigung -0,0997")

kopf("Aufgabe 1.1 -- die Verzerrung nachrechnen")
print(f"Cov(x,u) = 6 * (-2) = -12, Var(x) = 36 + 9 = 45, Verzerrung -12/45 = {-12 / 45:.3f}; -0,100 + ({-12 / 45:.3f}) = {-0.1 - 12 / 45:.3f}")
print(f"simuliert {steigung(y, x):.3f} -- der Rest stammt aus der Beschränkung auf nichtnegative Besuchszahlen; Cov(x,u) in den Daten: {np.cov(x, y - (-0.1) * x)[0, 1]:.2f}")
buch("-0,367 gegen -0,376")
