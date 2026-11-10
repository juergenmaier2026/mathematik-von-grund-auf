"""Band 27, Kapitel 11 -- Was hilft, und wie selten es schlimm wird.

Erzeugt: die vier Eingriffe (11.1), die Näherung von Sakasegawa (Satz 11.1),
den Spielerruin und die Aufspaltung (11.2, Definition 11.2) und die Rechnungen
aus den Aufgaben 11.1 bis 11.3.
Laufzeit unter zehn Sekunden.
"""
import _pfad
import math
import numpy as np
from mvg.station import Intensivstation
from mvg.erlang import sakasegawa
from mvg.aufspaltung import aufspaltung
from mvg.ausgabe import kopf, zeile, buch

STARTWERT = 2711
S = Intensivstation().mittlere_verweildauer          # 4,79 Tage
cv2 = math.exp(0.6 ** 2) - 1                          # quadrierter Variationskoeffizient der Lognormalen

kopf("Vier Eingriffe, je zehn Läufe à 20 000 Tage (11.1) -- und die Näherung von Sakasegawa (Satz 11.1)")
zeile("Variante", "rho", "wartet", "Wartezeit", "+-", "95 %", "Formel", breite=12)
for name, betten, faktor in (("12 Betten", 12, 1.0), ("13 Betten", 13, 1.0), ("14 Betten", 14, 1.0),
                             ("12, Dauer -10%", 12, 0.9), ("12, Dauer -20%", 12, 0.8)):
    st = Intensivstation(betten, los_faktor=faktor, warteschlange=True)
    laeufe = st.wiederholungen(10, 20_000, einschwingen=100, seed=STARTWERT + betten + int(faktor * 10))
    w = np.array([l["wartezeiten"].mean() for l in laeufe])
    pw = np.array([(l["wartezeiten"] > 0).mean() for l in laeufe])
    w95 = np.array([np.percentile(l["wartezeiten"], 95) for l in laeufe])
    zeile(name, round(st.last / betten, 3), round(pw.mean(), 2), round(w.mean(), 2), round(w.std(ddof=1) / math.sqrt(10), 2),
          round(w95.mean(), 1), round(sakasegawa(2.4, S * faktor, betten, cv2), 2), breite=12)
buch("Wartezeiten 5,5 / 1,36 / 0,56 / 1,06 / 0,30; Formel 5,7 / 1,35 / 0,57 / 1,02 / 0,33")

kopf("Spielerruin: Schlange von 1 auf 60, bevor sie leer ist -- 13 Betten, exponentielle Verweildauer (11.2)")
c, lam = 13, 2.4
mrate = 1 / S
p = lam / (lam + c * mrate)                           # Schritt nach oben in der eingebetteten Kette
r = (1 - p) / p
exakt = (1 - r) / (1 - r ** 60)
print(f"p = {p:.3f}, r = {r:.3f}, P(60 vor 0 | 1) = {exakt:.2e}")
buch("8,1e-5")

kopf("Aufspaltung (Definition 11.2): Schwellen 10 .. 60, N0 = 2000, R = 5, zwanzig Wiederholungen")
def lauf_bis(zustaende, schwelle, rng):
    q = zustaende.copy(); lebt = np.ones(len(q), bool); hit = np.zeros(len(q), bool)
    while lebt.any():
        idx = np.where(lebt)[0]
        q[idx] += np.where(rng.random(len(idx)) < p, 1, -1)
        oben, unten = q[idx] >= schwelle, q[idx] <= 0
        hit[idx[oben]] = True; lebt[idx[oben | unten]] = False
    return hit
rng = np.random.default_rng(STARTWERT)
erg = [aufspaltung(lauf_bis, range(10, 61, 10), 2000, 5, 1, rng) for _ in range(20)]
sch = np.array([e[0] for e in erg]); pfade = np.mean([e[2] for e in erg])
zeile("Schwelle", "Anteil exakt", "Anteil 1. Lauf", breite=16)
for i, l in enumerate(range(10, 61, 10)):
    ex = (1 - r ** (l - 10 if l > 10 else 1)) / (1 - r ** l)
    zeile(l, round(ex, 3), round(erg[0][1][i], 3), breite=16)
rel = sch.std(ddof=1) / sch.mean()
print(f"Schätzung ({sch.mean():.2e} +- {sch.std(ddof=1) / math.sqrt(20):.1e}), exakt {exakt:.2e}; Pfade je Lauf {pfade:,.0f};"
      f" relativer Fehler je Lauf {rel:.0%}; naiv für denselben Fehler: {1 / (exakt * rel ** 2):,.0f} Pfade")
buch("(8,2 +- 0,2)e-5, 9 600 Pfade, 12 %, naiv 810 000 -- Faktor 84; der Faktor schwankt mit dem Startwert zwischen 50 und 100")

kopf("Aufgabe 11.1 -- Betten für ein 95-%-Quantil unter zwei Tagen")
for betten in (14, 15, 16):
    wq = sakasegawa(2.4, S, betten, cv2)
    print(f"{betten} Betten: rho {2.4 * S / betten:.3f}, Wartezeit {wq:.2f} Tage, Quantil etwa 5,5 * {wq:.2f} = {5.5 * wq:.1f}")
buch("15 Betten")

kopf("Aufgabe 11.3 -- Häufigkeit je Jahr")
print(f"13 Betten: 130 Zyklen/Jahr * {exakt:.1e} = {130 * exakt:.3f} je Jahr -- einmal in hundert Jahren")
r12 = 1 / 0.958; p12 = (1 - r12) / (1 - r12 ** 60)
print(f"12 Betten: r = {r12:.3f}, P = {p12:.4f}, 100 Zyklen/Jahr -> {100 * p12:.2f} je Jahr -- alle drei Jahre")
buch("0,011 und 0,36 je Jahr")
