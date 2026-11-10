"""Band 22, Kapitel 3 -- Prüfen.

Erzeugt: das Leck (3.3) -- die größte Korrelation unter m Zufallsmerkmalen bei
n = 50 Patienten und was eine Kreuzvalidierung nach der Vorauswahl verspricht.
"""
import _pfad
import math
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2203)
n = 50

kopf("Größte |Korrelation| unter m Merkmalen aus reinem Zufall, n = 50, Mittel aus 200 Wiederholungen (3.3)")
zeile("m", "max |r| gemessen", "Näherung sqrt(2 ln m / n)", breite=26)
for m in (10, 100, 1000, 5000):
    werte = []
    for _ in range(200):
        y = rng.choice([-1.0, 1.0], n); X = rng.standard_normal((n, m))
        r = (X - X.mean(0)).T @ (y - y.mean()) / (n * X.std(0) * y.std())
        werte.append(np.abs(r).max())
    zeile(m, round(float(np.mean(werte)), 2), round(math.sqrt(2 * math.log(m) / n), 2), breite=26)
buch("0,30 / 0,43 / 0,53 / 0,58 Näherung; gemessen bei 5000 im Mittel 0,52")

kopf("Das Leck: zehn Merkmale an ALLEN Daten vorausgewählt, dann kreuzvalidiert -- gegen richtig verschachtelt")
def genauigkeit_mit_leck(X, y, k=10, falten=5):
    r = np.abs((X - X.mean(0)).T @ (y - y.mean())); top = np.argsort(r)[-k:]      # Auswahl sieht die Testfalten
    return kreuzvalidiert(X[:, top], y, falten)
def genauigkeit_richtig(X, y, k=10, falten=5):
    idx = np.arange(len(y)) % falten; treffer = 0
    for f in range(falten):
        tr, te = idx != f, idx == f
        r = np.abs((X[tr] - X[tr].mean(0)).T @ (y[tr] - y[tr].mean())); top = np.argsort(r)[-k:]   # nur Trainingsfalten
        treffer += (vorhersage(X[tr][:, top], y[tr], X[te][:, top]) == y[te]).sum()
    return treffer / len(y)
def vorhersage(Xtr, ytr, Xte):
    mp, mm = Xtr[ytr > 0].mean(0), Xtr[ytr < 0].mean(0)                    # nächstes Klassenmittel
    return np.where(((Xte - mp) ** 2).sum(1) < ((Xte - mm) ** 2).sum(1), 1.0, -1.0)
def kreuzvalidiert(X, y, falten=5):
    idx = np.arange(len(y)) % falten; treffer = 0
    for f in range(falten):
        tr, te = idx != f, idx == f
        treffer += (vorhersage(X[tr], y[tr], X[te]) == y[te]).sum()
    return treffer / len(y)
leck, richtig = [], []
for _ in range(100):
    y = rng.choice([-1.0, 1.0], n); X = rng.standard_normal((n, 5000))
    leck.append(genauigkeit_mit_leck(X, y)); richtig.append(genauigkeit_richtig(X, y))
print(f"scheinbare Trefferquote mit Leck {np.mean(leck):.0%}, richtig verschachtelt {np.mean(richtig):.0%} -- wahr: 50 %")
buch("der wahre Fehler jedes Verfahrens beträgt fünfzig Prozent; das Leck verspricht deutlich mehr")
