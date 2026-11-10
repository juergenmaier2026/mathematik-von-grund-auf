"""Band 26, Kapitel 9 -- Die doppelte Differenz.

Erzeugt: Reform in Jahr 3 mit wachsender Wirkung -- Vorperiodentest und
Ereignisstudie (9.3), und die Verzerrung der einfachen Zwei-mal-zwei-Differenz
bei wachsender Wirkung.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import kq
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2609)
n, T = 5000, 6
alpha = rng.standard_normal(n); behandelt = rng.random(n) < 0.5
jahre = np.arange(1, T + 1)
delta = 0.3 * jahre                                                          # gemeinsamer Trend
wirkung = np.where(jahre >= 3, jahre - 2, 0)                                 # 1, 2, 3, 4 ab Jahr 3
y = alpha[:, None] + delta[None, :] + behandelt[:, None] * wirkung[None, :] + rng.standard_normal((n, T))

kopf("Ereignisstudie: Reform in Jahr 3, Wirkung wächst um eins je Jahr (9.3)")
dm = lambda j: y[behandelt, j - 1].mean() - y[~behandelt, j - 1].mean()     # Differenz behandelt - Kontrolle im Jahr j
basis = (dm(1) + dm(2)) / 2
zeile("Vergleich", "Schätzung", "Soll", breite=22)
zeile("Jahr 2 gegen 1 (vorher)", round(dm(2) - dm(1), 2), 0, breite=22)
zeile("Jahr 3 gegen 1-2", round(dm(3) - basis, 2), 1, breite=22)
zeile("Jahr 5 gegen 1-2", round(dm(5) - basis, 2), 3, breite=22)
buch("-0,04 / 0,92 / 3,03")

kopf("Zwei-mal-zwei-Differenz (vorher/nachher gemittelt) bei konstanter und bei wachsender Wirkung")
for name, w in (("konstant 2", np.where(jahre >= 3, 2, 0)), ("wachsend 1..4", wirkung)):
    yy = alpha[:, None] + delta[None, :] + behandelt[:, None] * w[None, :] + rng.standard_normal((n, T))
    nach = (jahre >= 3)
    dd = (yy[behandelt][:, nach].mean() - yy[behandelt][:, ~nach].mean()) - (yy[~behandelt][:, nach].mean() - yy[~behandelt][:, ~nach].mean())
    print(f"   Wirkung {name:14s}: doppelte Differenz {dd:.2f} (Mittel der wahren Wirkung nach der Reform: {w[nach].mean():.2f})")
buch("bei konstantem Effekt 2,04 bei wahrem Wert 2 -- bei wachsender Wirkung schätzt die Differenz den Mittelwert über die Nachjahre")
