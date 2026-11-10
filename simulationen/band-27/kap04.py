"""Band 27, Kapitel 4 -- Den Fehler richtig messen.

Erzeugt: naiver gegen richtiger Standardfehler (4.1), Autokorrelationszeit
(Satz 4.1), Blockmittel (Satz 4.2), unabhängige Wiederholungen, die
Einschwingphase (4.3) und gemeinsame Zufallszahlen für 12 gegen 13 Betten
(Satz 4.3). Aufgabe 4.1 ist die Rechnung aus den Autokorrelationen.
"""
import _pfad
import math
import numpy as np
from mvg.station import Intensivstation, gemeinsame_zufallszahlen
from mvg.erlang import erlang_b
from mvg.statistik import se_naiv, autokorrelation, tau_int, se_blockmittel
from mvg.ausgabe import kopf, zeile, buch

STARTWERT = 2708
station = Intensivstation(12)
exakt = erlang_b(station.last, 12)

kopf("Eine Simulation, fünftausend Tage (Abschnitt 4.1)")
r = station.lauf(5000, einschwingen=0, seed=STARTWERT, protokoll=True)
folge = r["abweisung_folge"]                    # 1 = abgewiesen, je Ankunft
n, p = len(folge), folge.mean()
print(f"{n} Ankünfte, Anteil {p:.4f}, naiver Standardfehler {se_naiv(folge):.4f}")
rho = autokorrelation(folge, 10)
zeile("Abstand", 1, 2, 5, 10, breite=9)
zeile("rho_k", round(rho[0], 3), round(rho[1], 3), round(rho[4], 3), round(rho[9], 3), breite=9)
buch("4.1: 12 018 Ankünfte, 0,1819, naiv 0,0035; rho = 0,37 / 0,22 / 0,04 / -0,01")

kopf("Satz 4.1 -- Autokorrelationszeit")
tau = tau_int(folge)
print(f"tau = {tau:.2f}, wirksames n = {n / tau:.0f}, Standardfehler {folge.std(ddof=1) / math.sqrt(n / tau):.4f}")
buch("tau 3,1, wirksames n 3 880, SE 0,0062 -- die Blockmittel unten zeigen, dass tau eher 2,5 ist; "
     "der Wert hängt davon ab, wo man die Summe der rho_k abbricht")

kopf("Satz 4.2 -- Blockmittel")
zeile("Blöcke", "Länge", "SE", breite=10)
for b in (10, 50, 200):
    zeile(b, n // b, round(se_blockmittel(folge, b), 4), breite=10)
zeile("naiv", "", round(se_naiv(folge), 4), breite=10)
buch("0,0072 / 0,0064 / 0,0058 gegen naiv 0,0035")

kopf("Unabhängige Wiederholungen: 40 Läufe à 1000 Tage")
laeufe = station.wiederholungen(40, 1000, einschwingen=0, seed=STARTWERT + 1)
a = np.array([x["anteil"] for x in laeufe])
print(f"Mittel {a.mean():.4f}, Standardfehler {a.std(ddof=1) / math.sqrt(40):.4f}")
buch("0,1799 +- 0,0020")

kopf("Die Einschwingphase: Abweisungsanteil je Zeitfenster, 1500 leer gestartete Läufe (4.3)")
fenster = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 60), (60, 80), (80, 100), (100, 200)]
treffer = np.zeros(200); ank = np.zeros(200)
for k in np.random.SeedSequence(STARTWERT + 2).spawn(1500):
    sa, sd = k.spawn(2)
    x = station.lauf(200, einschwingen=0, rng_ankunft=np.random.default_rng(sa),
                     rng_dauer=np.random.default_rng(sd), protokoll=True)
    treffer += x["abweisungen_je_tag"]
    # Ankünfte je Tag sind nicht protokolliert; der Anteil wird je Fenster über die
    # erwartete Zahl 2,4 Ankünfte je Tag gebildet -- bei 1500 Läufen genau genug.
ank[:] = 2.4 * 1500
zeile("Tage", "Anteil", breite=10)
for lo, hi in fenster:
    zeile(f"{lo}-{hi}", round(treffer[lo:hi].sum() / ank[lo:hi].sum(), 3), breite=10)
buch("0-10: 0,101; 10-20: 0,174; danach 0,178-0,188 -- ab Tag 20 flach; Faustregel 5 * 4,8 = 24 Tage")

kopf("Satz 4.3 -- Zwölf gegen dreizehn Betten, je 30 Läufe à 2000 Tage, 50 Tage Einschwingphase")
s12, s13 = Intensivstation(12), Intensivstation(13)
unabh = np.array([s12.wiederholungen(1, 2000, 50, seed=STARTWERT + 10 + k)[0]["anteil"]
                  - s13.wiederholungen(1, 2000, 50, seed=STARTWERT + 50 + k)[0]["anteil"] for k in range(30)])
gemein = []
for k in range(30):
    r12, r13 = gemeinsame_zufallszahlen([s12, s13], 2000, 50, seed=STARTWERT + 100 + k)
    gemein.append(r12["anteil"] - r13["anteil"])
gemein = np.array(gemein)
zeile("Zufallszahlen", "Differenz", "SE", breite=14)
zeile("unabhängig", round(unabh.mean(), 4), round(unabh.std(ddof=1) / math.sqrt(30), 4))
zeile("gemeinsam", round(gemein.mean(), 4), round(gemein.std(ddof=1) / math.sqrt(30), 4))
zeile("exakt", round(exakt - erlang_b(s13.last, 13), 4), "")
print(f"Varianzverhältnis {unabh.var(ddof=1) / gemein.var(ddof=1):.1f}")
buch("unabhängig 0,0470 +- 0,0026; gemeinsam 0,0431 +- 0,0014; exakt 0,0422. "
     "Mit getrennten Strömen (Aufgabe 4.3) ist die Synchronität vollkommen und der Gewinn größer als im Buch")

kopf("Aufgabe 4.1 -- tau aus den Autokorrelationen 0,37 0,22 0,12 0,07 0,04")
tau_a = 1 + 2 * (0.37 + 0.22 + 0.12 + 0.07 + 0.04)
print(f"tau = {tau_a:.2f}, wirksames n = {12018 / tau_a:.0f}, SE = {0.386 / math.sqrt(12018 / tau_a):.4f}")
buch("2,64 / 4 552 / 0,0057")
