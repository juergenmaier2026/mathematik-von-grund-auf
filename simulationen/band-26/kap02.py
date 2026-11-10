"""Band 26, Kapitel 2 -- Identifikation.

Erzeugt: Angebot und Nachfrage -- die geschätzte Gerade als gewogenes Mittel der
beiden Steigungen (Satz 2.1), Formel gegen 400 000 Beobachtungen.
"""
import _pfad
import numpy as np
from mvg.oekonometrie import steigung
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2602)
a, b, c, d = 10.0, 0.8, 2.0, 0.6                      # Nachfrage q = a - b p + u, Angebot q = c + d p + v

kopf("Satz 2.1 -- was die Gerade zeigt: -b su^2/(su^2+sv^2) + d ... (400 000 Beobachtungen)")
zeile("sigma_u", "sigma_v", "Formel", "simuliert", breite=12)
for su, sv in ((1, 1), (1, 2), (2, 1), (0.2, 1), (1, 0.2)):
    u = rng.normal(0, su, 400_000); v = rng.normal(0, sv, 400_000)
    p = (a - c + u - v) / (b + d); q = a - b * p + u
    formel = (-b * sv ** 2 + d * su ** 2) / (su ** 2 + sv ** 2)
    zeile(su, sv, round(formel, 3), round(steigung(q, p), 3), breite=12)
buch("-0,100 / -0,520 / +0,320 / -0,746 / +0,546 -- die Gerade liegt zwischen -0,8 und +0,6")
