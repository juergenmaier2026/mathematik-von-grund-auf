"""Aufspaltung für seltene Ereignisse (Band 27, Kapitel 11)."""
import numpy as np


def aufspaltung(lauf_bis, schwellen, N0, R, start, rng):
    """Definition 11.2.

    lauf_bis(zustaende, schwelle, rng) -> Feld bool: erreicht jeder Startzustand die
                                          Schwelle, bevor er auf null fällt?
    schwellen   aufsteigende Liste l_1 < ... < l_m
    N0, R       Anfangspfade und Kopien je Treffer
    start       Startzustand (Skalar) der ersten Stufe
    Rückgabe: Schätzung, Liste der Anteile je Stufe, Gesamtzahl der Pfade.
    """
    zustaende = np.full(N0, start, float)
    schaetz, anteile, pfade = 1.0, [], N0
    for l in schwellen:
        treffer = lauf_bis(zustaende, l, rng)
        anteil = treffer.mean()
        anteile.append(anteil)
        schaetz *= anteil
        n = int(treffer.sum())
        if n == 0:
            return 0.0, anteile, pfade
        zustaende = np.full(n * R, l, float)      # Eintrittszustand: hier nur die Schlangenlänge (Markow-Fall)
        pfade += n * R
    return schaetz, anteile, pfade
