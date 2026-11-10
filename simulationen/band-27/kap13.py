"""Band 27, Kapitel 13 -- Dasselbe noch einmal.

Erzeugt: die drei Stufen der Reproduzierbarkeit (13.1), abgeleitete Ströme
(13.2), die Prüfleiter mit zwei eingebauten Fehlern (Satz 13.1), den Bruchfall
der Gleitkomma-Summe (13.5) und die Rechnung aus Aufgabe 13.1.
"""
import _pfad
import heapq
import math
import numpy as np
from mvg.station import Intensivstation
from mvg.erlang import erlang_b
from mvg.ausgabe import kopf, zeile, buch

station = Intensivstation(12)
exakt = erlang_b(station.last, 12)

kopf("Bitgenau, statistisch, sachlich (13.1)")
a = station.lauf(5000, seed=42)["anteil"]; b = station.lauf(5000, seed=42)["anteil"]; c = station.lauf(5000, seed=43)["anteil"]
print(f"Startwert 42 zweimal: {a:.12f} und {b:.12f} -> bitgenau gleich: {a == b}")
print(f"Startwert 43: {c:.4f}; exakt {exakt:.4f}; beide innerhalb von zwei Standardfehlern (0,006)")
buch("0,1861 / 0,1678 (andere Arbeitsfassung, anderer Generatorzustand)")

kopf("Abgeleitete Ströme (13.2): SeedSequence(2026).spawn(4)")
kinder = np.random.SeedSequence(2026).spawn(4)
print("erste Zahl jedes Kindes:", [int(np.random.default_rng(k).integers(10 ** 6)) for k in kinder])
print(f"Überlappungsrisiko bei einem Generator mit Periode 2^31, 40 Läufe à 10^6 Zahlen: etwa {40 ** 2 * 1e6 / 2 ** 31:.1f}")
buch("0,7 -- bei PCG64 mit Periode 2^128 praktisch null")

kopf("Satz 13.1 -- die Prüfleiter: zwei Fehler, die die Buchhaltung bestehen (10 Läufe à 5000 Tage)")
def station_mit_fehler(seed, fehler=None, tage=5000, burn=100):
    """Bewusst vereinfachte Fassung der Station, in die sich zwei Fehler einbauen lassen."""
    rng = np.random.default_rng(seed)
    t, belegt, ank, abw = 0.0, [], 0, 0
    while True:
        t += rng.exponential(1 / 2.4)
        if t > tage: break
        if fehler != "spaet":                             # richtig: Entlassungen VOR der Entscheidung
            while belegt and belegt[0] <= t: heapq.heappop(belegt)
        dauer = rng.lognormal(math.log(4), 0.6)
        if t > burn: ank += 1
        voll = len(belegt) >= (11 if fehler == "elf" else 12)
        if not voll: heapq.heappush(belegt, t + dauer)
        elif t > burn: abw += 1
        if fehler == "spaet":                             # falsch: Entlassungen NACH der Entscheidung
            while belegt and belegt[0] <= t: heapq.heappop(belegt)
    return abw / ank
zeile("Programm", "Anteil", "+-", "Urteil", breite=34)
for name, f in (("richtig", None), ("'voll' bei 11 statt 12", "elf"), ("Entlassungen nach der Aufnahme", "spaet")):
    r = np.array([station_mit_fehler(s, f) for s in range(10)])
    m, se = r.mean(), r.std(ddof=1) / math.sqrt(10)
    zeile(name, round(m, 4), round(se, 4), "ok" if abs(m - exakt) < 3 * se else "Stufe 3: Formel verletzt", breite=34)
zeile("exakt", round(exakt, 4), "", "", breite=34)
buch("0,1826 / 0,2308 / 0,2191 -- beide Fehler bestehen Erhaltung und Grenzfälle")

kopf("Der Bruchfall: Gleitkomma ist nicht assoziativ (13.5)")
x = np.random.default_rng(0).random(10 ** 6)
s1 = 0.0
for v in x: s1 += v
s2 = 0.0
for v in x[::-1]: s2 += v
print(f"vorwärts  {float(s1)!r}\nrückwärts {float(s2)!r}\nfsum      {math.fsum(x)!r}\nDifferenz {s1 - s2:.2e}")
buch("Unterschied in der zwölften Stelle -- harmlos, bis eine Verzweigung davon abhängt")

kopf("Aufgabe 13.1 -- ist das noch Zufall?")
for wert in (0.1861, 0.1678):
    print(f"{wert}: {(wert - 0.1785) / 0.006:+.1f} Standardfehler")
print(f"Differenz 0,0183 bei SE {0.006 * math.sqrt(2):.4f}: {0.0183 / (0.006 * math.sqrt(2)):.1f} Standardfehler -- kein Befund, kein Freispruch")
buch("1,3 / -1,8 / 2,2")
