"""Band 26, Kapitel 5 -- Ja oder nein, null oder mehr.

Erzeugt: Logit mit Randeffekten (Satz 5.2) und Ausgaben mit Nullen (Tobit, 5.3),
je 100 000 Versicherte. Die Parameter der latenten Modelle nennt das Buch nicht;
sie sind hier so gewählt, dass 21 % zum Facharzt gehen und 32 % keine Kosten haben.
"""
import _pfad
import numpy as np
from scipy.optimize import minimize
from mvg.versicherte import querschnitt
from mvg.oekonometrie import kq, steigung
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2605)
n = 100_000
g, x, _ = querschnitt(rng, n)
alter = rng.normal(50, 15, n)

kopf("Logit: Facharztbesuch ja/nein, Koeffizient -0,0374 je Euro (5.2)")
beta_wahr = np.array([-1.0, 0.02, -0.0374])                       # Konstante, Alter-50, Zuzahlung
X = np.column_stack([np.ones(n), alter - 50, x])
p_wahr = 1 / (1 + np.exp(-X @ beta_wahr))
y = (rng.random(n) < p_wahr).astype(float)
print(f"Anteil, der geht: {y.mean():.1%}")
nll = lambda b: -np.sum(y * (X @ b) - np.log1p(np.exp(X @ b)))
b = minimize(nll, np.zeros(3), method="BFGS").x
p = 1 / (1 + np.exp(-X @ b))
zeile("Größe", "je Euro Zuzahlung", breite=26)
zeile("Koeffizient beta", round(b[2], 4), breite=26)
zeile("Randeffekt bei p = 0,5", round(b[2] * 0.25, 4), breite=26)
zeile("mittlerer Randeffekt", round(float(np.mean(b[2] * p * (1 - p))), 4), breite=26)
zeile("lineares Modell", round(kq(y, alter, x)[0][2], 4), breite=26)
buch("-0,0374 / -0,0093 / -0,0059 / -0,0059")

kopf("Ausgaben mit Nullen: latente Neigung y* = 35 - 2 x + e, Kosten = max(0, y*) (5.3)")
ystern = 35 - 2 * x + rng.normal(0, 30, n)
kosten = np.maximum(0, ystern)
print(f"Anteil ohne Kosten: {np.mean(kosten == 0):.0%}")
zeile("Verfahren", "je Euro", breite=26)
zeile("KQ, alle Versicherten", round(steigung(kosten, x), 2), breite=26)
zeile("KQ, nur Zahler", round(steigung(kosten[kosten > 0], x[kosten > 0]), 2), breite=26)
zeile("wahre Steigung der Neigung", -2.0, breite=26)
buch("-1,38 / -1,00 / -2,00 -- beide Regressionen dämpfen, aus verschiedenen Gründen")
