"""Band 24, Kapitel 9 -- Banditen.

Erzeugt: vier Strategien für zwei Mittel mit Erfolgsraten 0,45 und 0,55, 1000
Behandlungen, 500 Wiederholungen (9.2): mittlere Reue und ihr Anteil an der Reue des Zufalls;
und die Reue bei bekannter Antwort unter epsilon = 0,1 (Aufgabe 9.1).
"""
import _pfad
import math
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2409)
p = np.array([0.45, 0.55]); T = 1000

def lauf(strategie):
    erfolge = np.zeros(2); versuche = np.zeros(2); schlecht = 0
    for t in range(T):
        if strategie == "zufällig": k = rng.integers(2)
        elif strategie == "epsilon":
            k = rng.integers(2) if rng.random() < 0.1 or versuche.min() == 0 else int(np.argmax(erfolge / versuche))
        elif strategie == "optimistisch":
            k = int(np.argmax(np.where(versuche > 0, erfolge / np.maximum(versuche, 1) + np.sqrt(2 * math.log(t + 1) / np.maximum(versuche, 1)), np.inf)))
        else:                                                   # Nachziehen (Thompson)
            k = int(np.argmax(rng.beta(1 + erfolge, 1 + versuche - erfolge)))
        erfolg = rng.random() < p[k]
        erfolge[k] += erfolg; versuche[k] += 1; schlecht += (k == 0)
    return T * p.max() - erfolge.sum(), schlecht / T

kopf("Zwei Mittel 0,45 und 0,55, 1000 Behandlungen, 500 Wiederholungen (9.2)")
zeile("Strategie", "mittlere Reue", "Anteil (zu zufällig)", breite=22)
reue_zufall = None
for name, s in (("zufällig (50/50)", "zufällig"), ("epsilon = 0,1", "epsilon"), ("optimistisch (UCB)", "optimistisch"), ("Nachziehen", "thompson")):
    r = np.array([lauf(s) for _ in range(500)])
    if reue_zufall is None: reue_zufall = r[:, 0].mean()
    zeile(name, round(float(r[:, 0].mean()), 1), f"{r[:, 0].mean() / reue_zufall:.0%}", breite=22)
buch("50,0 / 100 %; 18,6 / 37 %; 24,9 / 50 %; 12,3 / 25 %")

kopf("Aufgabe 9.1 -- Reue bei epsilon = 0,1, wenn das bessere Mittel schon bekannt ist")
print(f"zehn Prozent der Züge zufällig, davon die Hälfte auf das schlechtere: 1000 * 0,1 * 0,5 * 0,10 = {T * 0.1 * 0.5 * 0.1:.0f}")
buch("5 -- zwei Drittel der 18,6 entstehen, bevor das Verfahren weiß, welches Mittel besser ist")
