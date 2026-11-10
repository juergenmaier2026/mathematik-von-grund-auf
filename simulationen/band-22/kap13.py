"""Band 22, Kapitel 13 -- Hauptkomponenten.

Erzeugt: wie viele Komponenten 80, 90, 95 Prozent der Streuung erfassen --
Theorie für p = 20 gleichkorrelierte Merkmale mit rho = 0,7 gegen die Schätzung
aus 400 Patienten (Aufgabe 13.2).
"""
import _pfad
import numpy as np
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2213)
p, rho, n = 20, 0.7, 400
Sigma = np.full((p, p), rho); np.fill_diagonal(Sigma, 1.0)
ew_theorie = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
X = rng.multivariate_normal(np.zeros(p), Sigma, n)
ew = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
kopf("Komponenten für 80 / 90 / 95 Prozent der Streuung: Theorie gegen 400 Patienten (Aufgabe 13.2)")
zeile("Ziel", "Theorie", "geschätzt", breite=12)
for ziel in (0.8, 0.9, 0.95):
    k_t = int(np.argmax(np.cumsum(ew_theorie) / p >= ziel) + 1)
    k_s = int(np.argmax(np.cumsum(ew) / ew.sum() >= ziel) + 1)
    zeile(f"{ziel:.0%}", k_t, k_s, breite=12)
print(f"Eigenwerte: Theorie {ew_theorie[0]:.1f} und 19-mal {ew_theorie[1]:.1f}; geschätzt {ew[0]:.1f}, dann von {ew[1]:.2f} bis {ew[-1]:.2f}")
buch("7 / 14 / 17; geschätzt bei 90 % zwölf -- die kleinen Eigenwerte streuen ungleichmäßig")
