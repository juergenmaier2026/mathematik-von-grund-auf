"""Band 22, Kapitel 4 -- Regularisierung.

Erzeugt: kleinste Quadrate, Kammregression und Lasso an 300 Wiederholungen (4.4);
Strafstärke durch fünffache Kreuzvalidierung, geprüft an 2000 neuen Patienten.
Daten (vom Buch nicht genannt): n = 100, 50 Merkmale, fünf wirken mit Koeffizient 1, Rauschen 1.
"""
import _pfad
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2204)
n, p, wirksam = 100, 50, 5
beta = np.zeros(p); beta[:wirksam] = 1.0
fehler = {"kleinste Quadrate": [], "Kammregression": [], "Lasso": []}
for _ in range(300):
    X = rng.standard_normal((n, p)); y = X @ beta + rng.standard_normal(n)
    Xt = rng.standard_normal((2000, p)); yt = Xt @ beta + rng.standard_normal(2000)
    for name, modell in (("kleinste Quadrate", LinearRegression()),
                         ("Kammregression", RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5)),
                         ("Lasso", LassoCV(cv=5, random_state=0))):
        fehler[name].append(np.mean((yt - modell.fit(X, y).predict(Xt)) ** 2))
kopf("Dreihundert Wiederholungen (4.4)")
zeile("Verfahren", "Testfehler", "über Rauschen", breite=20)
for name, f in fehler.items():
    zeile(name, round(float(np.mean(f)), 2), round(float(np.mean(f)) - 1, 2), breite=20)
zeile("bestmöglich", 1.0, 0.0, breite=20)
buch("2,19 / 1,35 / 1,31 / 1,00 -- Lasso gewinnt, wenn wenige Merkmale wirken")
