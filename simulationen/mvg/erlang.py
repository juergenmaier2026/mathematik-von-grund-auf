"""Erlang-Formeln für Verlust- und Wartesysteme (Band 25, Kapitel 11; Band 27, Kapitel 1 und 11).

a   angebotene Last = Ankunftsrate * mittlere Verweildauer
c   Zahl der Betten (Bediener)
"""
import math


def erlang_b(a, c):
    """Abweisungswahrscheinlichkeit im Verlustsystem M/G/c/c (Erlang B).

    Rekursion B_0 = 1, B_k = a B_{k-1} / (k + a B_{k-1}); numerisch stabil,
    im Gegensatz zur Summenformel mit Fakultäten.
    """
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_b_tabelle(a, c_max):
    """Liste (k, B_k) für k = 0 .. c_max -- die Rekursion Schritt für Schritt."""
    b, zeilen = 1.0, [(0, 1.0)]
    for k in range(1, c_max + 1):
        b = a * b / (k + a * b)
        zeilen.append((k, b))
    return zeilen


def erlang_c(a, c):
    """Wartewahrscheinlichkeit im Wartesystem M/M/c (Erlang C). Verlangt a < c."""
    if a >= c:
        raise ValueError("Erlang C braucht a < c (Auslastung unter 1).")
    summe = sum(a ** k / math.factorial(k) for k in range(c))
    letzter = a ** c / math.factorial(c) * c / (c - a)
    return letzter / (summe + letzter)


def mittlere_wartezeit_mmc(lam, s, c):
    """Mittlere Wartezeit aller Ankünfte im M/M/c (Erlang C), s = mittlere Bedienzeit."""
    a = lam * s
    return erlang_c(a, c) / (c / s - lam)


def sakasegawa(lam, s, c, cv2):
    """Näherung von Sakasegawa (1977) für die mittlere Wartezeit im M/G/c.

    lam  Ankunftsrate, s mittlere Bedienzeit, c Bediener,
    cv2  quadrierter Variationskoeffizient der Bedienzeit.
    """
    rho = lam * s / c
    return rho ** (math.sqrt(2 * (c + 1)) - 1) / (c * (1 - rho)) * (1 + cv2) / 2 * s
