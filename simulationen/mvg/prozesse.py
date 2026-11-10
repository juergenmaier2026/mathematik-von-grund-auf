"""Stochastische Prozesse (Band 25): Irrfahrt, Pólya-Urne, Verzweigung, Brownsche Bewegung,
Ornstein-Uhlenbeck, Wright-Fisher, Nervenzelle."""
import numpy as np


def irrfahrten(rng, laeufe, schritte, d=1):
    """Einfache symmetrische Irrfahrt in d Dimensionen: Positionen (laeufe, schritte+1, d), ganzzahlig."""
    achse = rng.integers(0, d, (laeufe, schritte))
    vorz = rng.choice(np.array([-1, 1], np.int16), (laeufe, schritte))
    s = np.zeros((laeufe, schritte, d), np.int16)
    np.put_along_axis(s, achse[..., None], vorz[..., None], axis=2)
    return np.concatenate([np.zeros((laeufe, 1, d), np.int32), s.cumsum(axis=1, dtype=np.int32)], axis=1)


def polya_urne(rng, urnen, ziehungen):
    """Pólya-Urne, Start mit je einer roten und einer schwarzen Kugel; Endanteil rot."""
    rot = np.ones(urnen); gesamt = 2.0
    for _ in range(ziehungen):
        zieht_rot = rng.random(urnen) < rot / gesamt
        rot += zieht_rot; gesamt += 1
    return rot / gesamt


def wartezeit_muster(rng, muster, laeufe):
    """Mittlere Zahl der Münzwürfe, bis das Muster (z. B. 'KKZ') erstmals erscheint."""
    m = len(muster); zeiten = np.empty(laeufe)
    for i in range(laeufe):
        folge = ""; n = 0
        while not folge.endswith(muster):
            folge = (folge + ("K" if rng.random() < 0.5 else "Z"))[-m:]; n += 1
        zeiten[i] = n
    return zeiten.mean()


def verzweigung_aussterben(rng, nachkommen_summe, generationen, laeufe, start=1):
    """Anteil der Linien, die bis `generationen` ausgestorben sind.

    nachkommen_summe(rng, z) zieht für ein Feld z von Generationsgrößen die Größe der
    nächsten Generation -- die Summe von z unabhängigen Nachkommenzahlen (für Poisson
    und negative Binomialverteilung in geschlossener Form).
    """
    z = np.full(laeufe, start, np.int64)
    obergrenze = 10 ** 6                       # ab hier gilt die Linie als ausgebrochen und wird nicht weiter gerechnet
    for _ in range(generationen):
        offen = (z > 0) & (z < obergrenze)
        if not offen.any(): break
        z[offen] = np.minimum(nachkommen_summe(rng, z[offen]), obergrenze)
    return (z == 0).mean()


def brownsche_pfade(rng, pfade, schritte, T=1.0):
    """Brownsche Bewegung auf [0, T]: Pfade (pfade, schritte+1) mit W_0 = 0."""
    dW = rng.standard_normal((pfade, schritte)) * np.sqrt(T / schritte)
    return np.concatenate([np.zeros((pfade, 1)), dW.cumsum(axis=1)], axis=1)


def ornstein_uhlenbeck(rng, theta, xbar, sigma, x0, dt, schritte):
    """Euler-Maruyama für dX = theta (xbar - X) dt + sigma dW."""
    x = np.empty(schritte + 1); x[0] = x0
    for k in range(schritte):
        x[k + 1] = x[k] + theta * (xbar - x[k]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return x


def wright_fisher(rng, N, p0, populationen, max_gen=100_000):
    """Genetische Drift: Anteil fixierter Populationen und mittlere Entscheidungszeit."""
    p = np.full(populationen, p0); zeit = np.zeros(populationen); offen = np.ones(populationen, bool)
    for g in range(max_gen):
        if not offen.any(): break
        p[offen] = rng.binomial(N, p[offen]) / N
        fertig = offen & ((p == 0) | (p == 1)); zeit[fertig] = g + 1; offen &= ~fertig
    return (p == 1).mean(), zeit.mean(), g + 1


def nervenzelle(rng, tau, vbar, sigma, schwelle, reset, dt, schritte, v0=None):
    """Integrate-and-fire: Ornstein-Uhlenbeck mit Schwelle; Rückgabe: Feuerzeitpunkte (ms)."""
    v = reset if v0 is None else v0; zeiten = []
    for k in range(schritte):
        v += (vbar - v) / tau * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        if v >= schwelle:
            zeiten.append(k * dt); v = reset
    return np.array(zeiten)
