"""Band 22, Kapitel 2 -- Überanpassung.

Erzeugt: Verzerrung und Varianz von Polynomen des Grades 0 bis 9 an 4000
simulierten Kliniken (2.3) und den Optimismus-Test (Satz 2.2); Daten:
L(t) = 1 + 3 exp(-t/4) plus Messfehler mit Streuung 0,3, zehn Messungen.
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2202)
t = np.linspace(0, 12, 10); sigma = 0.3
f = lambda t: 1 + 3 * np.exp(-t / 4)
t_test = np.linspace(0, 12, 200)

kopf("4000 Kliniken, je zehn Messungen: Training, Verzerrung^2, Varianz, Testfehler (2.3)")
zeile("Grad", "Training", "Verzerr.^2", "Varianz", "Test", breite=12)
for grad in (0, 1, 2, 3, 5, 7, 9):
    V = np.vander(t / 12, grad + 1); Vt = np.vander(t_test / 12, grad + 1)
    vorhersagen, training = [], []
    for _ in range(4000):
        y = f(t) + rng.normal(0, sigma, 10)
        b = np.linalg.lstsq(V, y, rcond=None)[0]
        training.append(np.mean((y - V @ b) ** 2)); vorhersagen.append(Vt @ b)
    P = np.array(vorhersagen)
    verz2 = np.mean((P.mean(axis=0) - f(t_test)) ** 2); var = np.mean(P.var(axis=0))
    zeile(grad, round(float(np.mean(training)), 3), round(float(verz2), 3), round(float(var), 3), round(float(verz2 + var + sigma ** 2), 3), breite=12)
buch("Grad 2 hat den kleinsten Testfehler 0,119; Grad 9: Training 0, Varianz 0,595")

kopf("Satz 2.2 -- Optimismus = 2 sigma^2 p / n, gemessen an 20 000 Wiederholungen")
zeile("n", "p", "gemessen", "2 s^2 p/n", breite=12)
for p in (2, 3, 6):
    V = np.vander(t / 12, p); opt = []
    for _ in range(20_000):
        y = f(t) + rng.normal(0, sigma, 10); b = np.linalg.lstsq(V, y, rcond=None)[0]
        y_neu = f(t) + rng.normal(0, sigma, 10)
        opt.append(np.mean((y_neu - V @ b) ** 2) - np.mean((y - V @ b) ** 2))
    zeile(10, p, round(float(np.mean(opt)), 4), round(2 * sigma ** 2 * p / 10, 4), breite=12)
buch("0,0366 / 0,0542 / 0,1072 gegen 0,036 / 0,054 / 0,108")
