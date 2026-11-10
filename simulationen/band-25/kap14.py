"""Band 25, Kapitel 14 -- Prozesse in Klinik und Natur.

Erzeugt: die Nervenzelle als Ornstein-Uhlenbeck-Prozess mit Schwelle (14.1) und
die genetische Drift nach Wright-Fisher (Satz 14.1): Fixierung und Zeit bis zur
Entscheidung, 20 000 Populationen.
"""
import _pfad
import numpy as np
from mvg.prozesse import nervenzelle, wright_fisher
from mvg.ausgabe import kopf, buch

rng = np.random.default_rng(2514)

kopf("Die Nervenzelle: tau = 20 ms, Reset -70 mV, Schwelle -50 mV, Antrieb -45,9 mV, sigma = 0,3 mV/sqrt(ms) (14.1)")
zeiten = nervenzelle(rng, tau=20.0, vbar=-45.9, sigma=0.3, schwelle=-50.0, reset=-70.0, dt=0.05, schritte=2_000_000)
isi = np.diff(zeiten)
print(f"Feuerrate {1000 / isi.mean():.1f} Hz, mittlerer Abstand {isi.mean():.1f} ms, Streuungsmaß (CV) {isi.std() / isi.mean():.3f}")
buch("28,2 Hz, 35,4 ms, 0,131 -- das Buch nennt nur tau; Antrieb und Rauschen sind hier so gewählt, dass seine Zahlen herauskommen: "
     "ohne Rauschen feuert die Zelle alle tau ln((vbar-reset)/(vbar-Schwelle)) = 35,4 ms")

kopf("Satz 14.1 -- genetische Drift: N = 1000, Start bei p = 0,1, 20 000 Populationen")
fix, zeit, gen = wright_fisher(rng, 1000, 0.1, 20_000)
print(f"Fixierung in {fix:.1%} der Fälle (Satz: 10 %), mittlere Entscheidungszeit {zeit:.0f} Generationen, letzte Entscheidung nach {gen} Generationen")
buch("10,1 Prozent; die letzte Entscheidung fällt je nach Startwert nach 10 000 bis 20 000 Generationen -- der lange Schwanz aus Aufgabe 14.2")
