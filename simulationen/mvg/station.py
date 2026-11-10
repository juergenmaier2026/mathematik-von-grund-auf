"""Die Intensivstation -- der Leitfall von Band 27 als ereignisgesteuerte Simulation
(Kapitel 1, 4, 5, 10, 11).

Modell
  Ankünfte       Poisson-Prozess mit Rate lam je Tag
  Verweildauer   logarithmisch normalverteilt, Median exp(mu), Streuung sigma
                 auf der logarithmischen Skala (Standard: Median 4 Tage, sigma 0,6)
  Betten         c; ist keines frei, wird der Patient abgewiesen
                 (oder er wartet, wenn warteschlange=True)

Zustand       die Entlassungszeiten der belegten Betten (eine Halde)
Ereignisse    Ankunft, Entlassung; Entlassungen vor der nächsten Ankunft
              werden verarbeitet, BEVOR über die Aufnahme entschieden wird
              (Reihenfolge-Regel, Kapitel 10 und 13)
Zufall        zwei getrennte Ströme -- Ankünfte und Verweildauern --, damit
              Varianten mit gemeinsamen Zufallszahlen synchron bleiben (Aufgabe 4.3).
              Die Verweildauer wird für JEDEN Patienten gezogen, auch für abgewiesene.
"""
import heapq
import math
import numpy as np


class Intensivstation:
    def __init__(self, betten=12, lam=2.4, median=4.0, sigma=0.6,
                 los_faktor=1.0, warteschlange=False):
        self.betten = betten
        self.lam = lam
        self.mu = math.log(median)
        self.sigma = sigma
        self.los_faktor = los_faktor
        self.warteschlange = warteschlange

    @property
    def mittlere_verweildauer(self):
        return self.los_faktor * math.exp(self.mu + self.sigma ** 2 / 2)

    @property
    def last(self):
        """angebotene Last a = lam * mittlere Verweildauer"""
        return self.lam * self.mittlere_verweildauer

    def _verweildauer(self, rng):
        return self.los_faktor * rng.lognormal(self.mu, self.sigma)

    def lauf(self, tage, einschwingen=0.0, seed=None, rng_ankunft=None, rng_dauer=None,
             protokoll=False):
        """Ein Lauf über `tage` Tage. Gezählt wird erst nach `einschwingen` Tagen.

        seed            ein Hauptstartwert; daraus werden die beiden Ströme abgeleitet
        rng_ankunft,    oder: die beiden Generatoren direkt übergeben
        rng_dauer       (für gemeinsame Zufallszahlen zweier Varianten)
        protokoll       zusätzlich die Folge der Abweisungen je Ankunft und
                        die Zahl der Abweisungen je Tag zurückgeben

        Rückgabe: dict mit
          ankuenfte, abweisungen, anteil       -- nach der Einschwingphase
          belegung                             -- Zeitmittel der belegten Betten
          mittlere_dauer                       -- Mittel der gezogenen Verweildauern
          wartezeiten (nur mit warteschlange)  -- je aufgenommener Patient
          schlange                             -- Zeitmittel der Schlangenlänge
          abweisung_folge, abweisungen_je_tag  -- nur mit protokoll
        """
        if rng_ankunft is None or rng_dauer is None:
            ss = np.random.SeedSequence(seed)
            k1, k2 = ss.spawn(2)
            rng_ankunft = np.random.default_rng(k1)
            rng_dauer = np.random.default_rng(k2)
        t = 0.0
        belegt = []            # Halde der Entlassungszeiten
        schlange = []          # Ankunftszeiten der Wartenden
        letzte = 0.0           # Zeitpunkt des letzten Ereignisses (für die Zeitmittel)
        bel_int = 0.0          # Integral der Belegung
        schl_int = 0.0         # Integral der Schlangenlänge
        ankuenfte = abweisungen = 0
        dauer_summe = 0.0
        wartezeiten = []
        folge = []
        je_tag = np.zeros(int(tage) + 1, int)

        def entlassen_bis(zeit):
            nonlocal letzte, bel_int, schl_int
            while belegt and belegt[0] <= zeit:
                td = heapq.heappop(belegt)
                bel_int += (td - letzte) * (len(belegt) + 1)
                schl_int += (td - letzte) * len(schlange)
                letzte = td
                if schlange:                      # Wartevariante: Nächsten aufnehmen
                    ta = schlange.pop(0)
                    if ta > einschwingen:
                        wartezeiten.append(td - ta)
                    heapq.heappush(belegt, td + self._verweildauer(rng_dauer))

        while True:
            t += rng_ankunft.exponential(1.0 / self.lam)
            if t > tage:
                entlassen_bis(tage)
                bel_int += (tage - letzte) * len(belegt)
                schl_int += (tage - letzte) * len(schlange)
                break
            entlassen_bis(t)                     # erst die fälligen Entlassungen ...
            bel_int += (t - letzte) * len(belegt)
            schl_int += (t - letzte) * len(schlange)
            letzte = t
            dauer = self._verweildauer(rng_dauer)   # ... dann die Ankunft
            zaehlen = t > einschwingen
            if zaehlen:
                ankuenfte += 1
                dauer_summe += dauer
            if len(belegt) < self.betten:
                heapq.heappush(belegt, t + dauer)
                if zaehlen:
                    if self.warteschlange:
                        wartezeiten.append(0.0)
                    if protokoll:
                        folge.append(0)
            elif self.warteschlange:
                schlange.append(t)
            else:
                if zaehlen:
                    abweisungen += 1
                    if protokoll:
                        folge.append(1)
                        je_tag[int(t)] += 1

        erg = dict(ankuenfte=ankuenfte, abweisungen=abweisungen,
                   anteil=abweisungen / ankuenfte if ankuenfte else float("nan"),
                   belegung=bel_int / tage,
                   mittlere_dauer=dauer_summe / ankuenfte if ankuenfte else float("nan"))
        if self.warteschlange:
            erg["wartezeiten"] = np.array(wartezeiten)
            erg["schlange"] = schl_int / tage
        if protokoll:
            erg["abweisung_folge"] = np.array(folge)
            erg["abweisungen_je_tag"] = je_tag[int(einschwingen):int(tage)]
        return erg

    def wiederholungen(self, laeufe, tage, einschwingen=0.0, seed=0, **kw):
        """`laeufe` unabhängige Läufe; Lauf k erhält den k-ten Kindstartwert von seed."""
        kinder = np.random.SeedSequence(seed).spawn(laeufe)
        ergebnisse = []
        for k in kinder:
            a, d = k.spawn(2)
            ergebnisse.append(self.lauf(tage, einschwingen,
                                        rng_ankunft=np.random.default_rng(a),
                                        rng_dauer=np.random.default_rng(d), **kw))
        return ergebnisse


def gemeinsame_zufallszahlen(stationen, tage, einschwingen=0.0, seed=0):
    """Dieselben Zufallszahlen für mehrere Varianten (Satz 4.3).

    Alle Stationen in `stationen` erhalten identische Generatoren; weil die
    Verweildauer für jeden Patienten gezogen wird, bleiben die Varianten synchron.
    """
    ss = np.random.SeedSequence(seed)
    a, d = ss.spawn(2)
    out = []
    for s in stationen:
        out.append(s.lauf(tage, einschwingen,
                          rng_ankunft=np.random.default_rng(a),
                          rng_dauer=np.random.default_rng(d)))
    return out
