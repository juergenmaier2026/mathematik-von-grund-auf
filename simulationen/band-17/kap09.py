"""Band 17, Kapitel 9 -- Verteilungsfreie Verfahren und Simulation.

Erzeugt: den Permutationstest an den MOBIL-Mobilisationszeiten (9.5, exakt über
alle 924 Aufteilungen und durch Ziehen), den Bootstrap für die Mittelwertdifferenz
und für den Median (9.6).
"""
import _pfad
import itertools
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(1709)
A = np.array([19, 22, 24, 26, 31, 35.0]); B = np.array([14, 15, 17, 18, 21, 23.0])
alle = np.concatenate([A, B]); beob = A.mean() - B.mean()

kopf("Permutationstest: Differenz der Mittelwerte, alle 924 Aufteilungen (9.5)")
treffer = 0
for idx in itertools.combinations(range(12), 6):
    a = alle[list(idx)]; b = np.delete(alle, list(idx))
    treffer += abs(a.mean() - b.mean()) >= abs(beob) - 1e-9
print(f"beobachtet {beob:.3f}; {treffer} von 924 Aufteilungen mindestens so extrem: p = {treffer / 924:.4f}")
gezogen = np.mean([abs((p := rng.permutation(alle))[:6].mean() - p[6:].mean()) >= abs(beob) - 1e-9 for _ in range(100_000)])
print(f"mit 100 000 zufälligen Aufteilungen: p = {gezogen:.4f}")
buch("14 / 924 = 0,0152; t-Test 0,0153")

kopf("Bootstrap, 20 000 Ziehungen mit Zurücklegen (9.6)")
diff = np.array([rng.choice(A, 6).mean() - rng.choice(B, 6).mean() for _ in range(20_000)])
med = np.array([np.median(rng.choice(B, 6)) for _ in range(20_000)])
zeile("Größe", "2,5 %", "97,5 %", breite=20)
zeile("Mittelwertdifferenz", round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2), breite=20)
zeile("Median Gruppe B", round(float(np.percentile(med, 2.5)), 1), round(float(np.percentile(med, 97.5)), 1), breite=20)
print(f"t-Intervall zum Vergleich: [1,93; 14,40]; Punktschätzer Median B: {np.median(B)}")
buch("[3,33; 13,33] und [14,5; 22,0] -- der Bootstrap kann nur Werte erzeugen, die aus der Stichprobe kommen")
