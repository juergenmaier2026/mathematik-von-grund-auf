"""Pfade stochastischer Differentialgleichungen (Band 27, Kapitel 9.3 und 9.4)."""
import numpy as np


def euler_maruyama(a, b, x0, T, n_schritte, dW):
    """Definition 9.3. dW: Zuwächse der Form (Pfade, n_schritte). Rückgabe: Endwerte."""
    dt = T / n_schritte
    x = np.full(dW.shape[0], x0, float)
    for k in range(n_schritte):
        x = x + a(x) * dt + b(x) * dW[:, k]
    return x


def milstein(a, b, db, x0, T, n_schritte, dW):
    """Milstein-Verfahren: Euler plus 1/2 b b' (dW^2 - dt)."""
    dt = T / n_schritte
    x = np.full(dW.shape[0], x0, float)
    for k in range(n_schritte):
        x = x + a(x) * dt + b(x) * dW[:, k] + 0.5 * b(x) * db(x) * (dW[:, k] ** 2 - dt)
    return x


def brownsche_zuwaechse(rng, pfade, feinste, T):
    """Zuwächse auf dem feinsten Gitter; vergröbern() summiert sie für gröbere Gitter,
    sodass alle Gitter denselben Brownschen Pfad sehen (gemeinsame Zufallszahlen)."""
    return rng.standard_normal((pfade, feinste)) * np.sqrt(T / feinste)


def vergroebern(dW, n_schritte):
    """Zuwächse des feinsten Gitters zu n_schritte Zuwächsen zusammenfassen."""
    pfade, feinste = dW.shape
    k = feinste // n_schritte
    return dW.reshape(pfade, n_schritte, k).sum(axis=2)


def mehrebenen_pfadzahlen(V, C, eps):
    """Satz 9.5: optimale Pfadzahlen N_l ~ sqrt(V_l / C_l) für Gesamtvarianz eps^2 / 2."""
    V, C = np.asarray(V, float), np.asarray(C, float)
    s = np.sum(np.sqrt(V * C))
    return np.ceil(2 / eps ** 2 * np.sqrt(V / C) * s).astype(int)
