"""Band 27, Kapitel 1 -- Ein Experiment im Rechner.

Erzeugt: die Erlang-Rekursion (Abschnitt 1.1, Aufgabe 1.1), die Simulation
über 100 bis 100 000 Tage (1.1), die fünfzig Wiederholungen (1.2) und die
Rechnung aus Aufgabe 1.2.
"""
import _pfad
import math
import numpy as np
from mvg.station import Intensivstation
from mvg.erlang import erlang_b, erlang_b_tabelle
from mvg.ausgabe import kopf, zeile, buch

STARTWERT = 2701

station = Intensivstation(betten=12, lam=2.4, median=4.0, sigma=0.6)
a = station.last
exakt = erlang_b(a, 12)

kopf("Die Formel von Erlang")
print(f"Last a = {station.lam} * {station.mittlere_verweildauer:.2f} = {a:.2f}")
print(f"Abweisungswahrscheinlichkeit bei 12 Betten: {exakt:.4f}")
buch("Abschnitt 1.1, exakt 0,1785")

kopf("Die Rekursion Schritt für Schritt (Aufgabe 1.1)")
zeile("Betten", "B_c", breite=10)
for k, b in erlang_b_tabelle(a, 13):
    zeile(k, round(b, 4), breite=10)
print(f"Dreizehntes Bett: {exakt:.4f} - {erlang_b(a, 13):.4f} = {exakt - erlang_b(a, 13):.4f}")
buch("Aufgabe 1.1: 0,1785 - 0,136 = 0,042")

kopf("Dasselbe als Simulation (Abschnitt 1.1)")
zeile("Tage", "Ankünfte", "Anteil", "naiver SE")
for tage in (100, 1000, 10_000, 100_000):
    r = station.lauf(tage, einschwingen=0, seed=STARTWERT)
    p, n = r["anteil"], r["ankuenfte"]
    zeile(tage, n, round(p, 3), round(math.sqrt(p * (1 - p) / n), 3))
zeile("exakt", "", round(exakt, 4), "")
buch("Tabelle in 1.1 -- Werte innerhalb des Standardfehlers; "
     "die erste Zeile liegt wegen der Einschwingphase zu niedrig")

kopf("Fünfzig Wiederholungen à 1000 Tage (Abschnitt 1.2)")
laeufe = station.wiederholungen(50, 1000, einschwingen=0, seed=STARTWERT + 10)
anteile = np.array([r["anteil"] for r in laeufe])
print(f"Mittel der 50 Läufe          {anteile.mean():.4f}")
print(f"Streuung der Einzelläufe     {anteile.std(ddof=1):.4f}")
print(f"Standardfehler des Mittels   {anteile.std(ddof=1) / math.sqrt(50):.4f}")
print(f"exakt                        {exakt:.4f}")
buch("1.2: Mittel 0,1797, Streuung 0,0128, Standardfehler 0,0018")

kopf("Aufgabe 1.2 -- Wie viele Läufe für +-0,002?")
sigma = math.sqrt(0.18 * 0.82)
n = (1.96 * sigma / 0.002) ** 2
print(f"sigma je Ankunft = {sigma:.3f};  n >= (1,96*sigma/0,002)^2 = {n:,.0f} Ankünfte"
      f" = {n / 2.4:,.0f} Tage (bei Unabhängigkeit)")
n_eff = (0.384 / anteile.std(ddof=1)) ** 2
print(f"Wirksame Zahl je 1000-Tage-Lauf aus der Streuung: (0,384/{anteile.std(ddof=1):.4f})^2 = {n_eff:,.0f}"
      f"  (statt {laeufe[0]['ankuenfte']} Ankünfte)")
buch("Aufgabe 1.2: 141 600 Ankünfte, 59 000 Tage; wirksame Zahl etwa 900")
