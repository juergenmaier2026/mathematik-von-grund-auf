"""Zufallszahlen aus Formeln (Band 27, Kapitel 2)."""
import numpy as np


def lcg(a, c, m, x0, n):
    """Die ersten n Werte des linearen Kongruenzgenerators x_{k+1} = (a x_k + c) mod m."""
    werte, x = [], x0
    for _ in range(n):
        werte.append(x)
        x = (a * x + c) % m
    return werte


def lcg_gleichverteilt(a, c, m, x0, n):
    """Wie lcg, aber als Zahlen in [0, 1): x_k / m."""
    return np.array(lcg(a, c, m, x0, n), float) / m


def randu(x0, n):
    """IBMs RANDU: x_{k+1} = 65539 x_k mod 2^31 (x0 ungerade), als Zahlen in [0, 1)."""
    return lcg_gleichverteilt(65539, 0, 2 ** 31, x0, n)


def inversion(F_inv, u):
    """Satz 2.4: X = F^{-1}(U)."""
    return F_inv(u)


def box_muller(rng, n):
    """Satz 2.5: n Paare standardnormalverteilter Zahlen aus 2n Gleichverteilten."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(-2.0 * np.log(u1))
    return r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)


def verwerfung(rng, ziehe_vorschlag, quotient, n):
    """Satz 2.6: Verwerfungsverfahren.

    ziehe_vorschlag(rng, k) liefert k Vorschläge aus der Hülle g,
    quotient(x) = f(x) / (M g(x)) in [0, 1].
    Rückgabe: n angenommene Werte und die Annahmequote.
    """
    angenommen, versuche = [], 0
    while len(angenommen) < n:
        k = n - len(angenommen)
        x = ziehe_vorschlag(rng, k)
        u = rng.random(k)
        versuche += k
        angenommen.extend(x[u <= quotient(x)])
    return np.array(angenommen[:n]), n / versuche
