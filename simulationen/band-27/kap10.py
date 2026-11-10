"""Band 27, Kapitel 10 -- Die Station als Programm.

Erzeugt: Zähler und Little-Probe (10.2), Zeitschritt gegen Ereignis (10.3),
die Wartevariante (10.4), die drei Ereignisse von Hand (Aufgabe 10.1) und
Little in der Wartevariante (Aufgabe 10.2).
Laufzeit wenige Sekunden.
"""
import _pfad
import heapq
import math
import numpy as np
from mvg.station import Intensivstation
from mvg.erlang import erlang_b, erlang_c, mittlere_wartezeit_mmc
from mvg.ausgabe import kopf, zeile, buch

STARTWERT = 2710
station = Intensivstation(12)
W = station.mittlere_verweildauer

kopf("Die Zähler und der Satz von Little (10.2), 20 000 Tage")
r = station.lauf(20_000, einschwingen=100, seed=STARTWERT)
print(f"Abweisungsanteil {r['anteil']:.4f}, mittlere Belegung (Zeitmittel) {r['belegung']:.2f}")
print(f"Little: lambda (1-p) W = 2,4 * {1 - r['anteil']:.4f} * {W:.2f} = {2.4 * (1 - r['anteil']) * W:.2f}")
buch("0,1795 und 9,45 gegen 9,43")

kopf("Zeitschritt statt Ereignis (10.3): Restdauern je Schritt vermindern, Poisson-Ankünfte je Schritt")
def zeitschritt(dt, tage=20_000, einschwingen=100, entlassungen_zuerst=True, seed=STARTWERT):
    rng = np.random.default_rng(seed)
    rest, ank, abw = [], 0, 0
    for k in range(int(tage / dt)):
        t = k * dt
        if entlassungen_zuerst:
            rest = [x - dt for x in rest if x - dt > 0]
        for _ in range(rng.poisson(2.4 * dt)):
            dauer = rng.lognormal(station.mu, station.sigma)
            if t > einschwingen: ank += 1
            if len(rest) < 12: rest.append(dauer)
            elif t > einschwingen: abw += 1
        if not entlassungen_zuerst:
            rest = [x - dt for x in rest if x - dt > 0]
    return abw / ank
zeile("Bauart", "dt", "Anteil", breite=18)
zeile("ereignisgesteuert", "---", round(r["anteil"], 4), breite=18)
for dt, name in ((1 / 24, "1 Stunde"), (0.25, "6 Stunden"), (1.0, "1 Tag")):
    zeile("Zeitschritt", name, round(zeitschritt(dt), 4), breite=18)
print(f"exakt {erlang_b(station.last, 12):.4f}; Standardfehler je Zeile etwa 0,004")
buch("Tagesschritte 0,193 -- um 0,014 zu hoch, weil jeder Patient einen halben Schritt zu lang liegt")
print(f"Ankünfte vor den Entlassungen verarbeitet (Bruchfall 10.5): {zeitschritt(1.0, entlassungen_zuerst=False):.4f} -- in dieser Bauart")
print("   exakt dasselbe, weil sich nur jeder Aufenthalt um einen Schritt verschiebt; die Reihenfolge-Frage stellt sich")
print("   erst bei Ereignissen auf demselben Raster, etwa Entlassungen zur Visite um acht Uhr.")
buch("10.5 nennt 0,193 gegen 0,194 aus unabhängigen Läufen -- der Unterschied dort ist Zufall")

kopf("Die Wartevariante (10.4): zehn Läufe à 20 000 Tage")
warte = Intensivstation(12, warteschlange=True)
laeufe = warte.wiederholungen(10, 20_000, einschwingen=100, seed=STARTWERT + 1)
pw = np.array([(l["wartezeiten"] > 0).mean() for l in laeufe])
w = np.array([l["wartezeiten"].mean() for l in laeufe])
w95 = np.array([np.percentile(l["wartezeiten"], 95) for l in laeufe])
Lq = np.array([l["schlange"] for l in laeufe])
se = lambda x: x.std(ddof=1) / math.sqrt(len(x))
a = station.last
zeile("", "Simulation", "+-", "Erlang C", breite=26)
zeile("Anteil, der wartet", round(pw.mean(), 3), round(se(pw), 3), round(erlang_c(a, 12), 3), breite=26)
zeile("mittlere Wartezeit (Tage)", round(w.mean(), 2), round(se(w), 2), round(mittlere_wartezeit_mmc(2.4, W, 12), 2), breite=26)
zeile("95-%-Quantil (Tage)", round(w95.mean(), 1), round(se(w95), 1), "", breite=26)
zeile("mittlere Schlangenlänge", round(Lq.mean(), 1), round(se(Lq), 1), "", breite=26)
print(f"Aufgabe 10.2, Little für die Schlange: lambda * W_q = 2,4 * {w.mean():.2f} = {2.4 * w.mean():.1f} gegen gezählt {Lq.mean():.1f}")
buch("0,83 / 5,5 +- 0,4 / 18 / 13 -- Erlang C trifft den Anteil, nicht die Wartezeit")

kopf("Aufgabe 10.1 -- drei Ereignisse von Hand (zwei Betten)")
uhr, liste, belegt = 0.0, [(0.4, "A"), (1.5, "E1")], [1.5]
ankuenfte = iter([(0.4, 2.0), (1.2, 3.0), (1.7, None)])
integral, letzte = 0.0, 0.0
naechste = next(ankuenfte)
while liste:
    zeit, art = heapq.heappop(liste)
    integral += (zeit - letzte) * len(belegt); letzte = zeit
    if art == "A":
        t, dauer = naechste
        if len(belegt) < 2 and dauer is not None:
            belegt.append(zeit + dauer); heapq.heappush(liste, (zeit + dauer, f"E{len(belegt)}")); was = f"aufgenommen, Entlassung bei {zeit + dauer}"
        elif dauer is None:
            was = "Ankunft, Bett frei -- Ende der Aufgabe"
        else:
            was = "abgewiesen (beide Betten belegt)"
        try:
            naechste = next(ankuenfte); heapq.heappush(liste, (naechste[0], "A"))
        except StopIteration:
            pass
    else:
        belegt.remove(zeit); was = "Bett frei"
    print(f"Uhr {zeit:4.1f}  {art:3s} {was:45s} Integral {integral:4.1f}  Liste {sorted(liste)}")
    if art == "A" and naechste[1] is None and zeit == 1.7: break
print(f"mittlere Belegung bis 1,7: {integral / 1.7:.2f}")
buch("Aufgabe 10.1: Integral 2,8, mittlere Belegung 1,65")
