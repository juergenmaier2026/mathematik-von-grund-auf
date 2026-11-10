"""Statistik der Simulationsausgaben (Band 27, Kapitel 4, 6 und 8)."""
import numpy as np


def se_naiv(x):
    """Standardfehler des Mittels unter Unabhängigkeit: s / sqrt(n)."""
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))


def autokorrelation(x, max_lag):
    """Empirische Autokorrelationen rho_1 .. rho_max_lag."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x) / n
    return np.array([np.dot(x[:n - k], x[k:]) / n / c0 for k in range(1, max_lag + 1)])


def tau_int(x, max_lag=None, schwelle=0.05):
    """Integrierte Autokorrelationszeit tau = 1 + 2 sum rho_k.

    Die Summe wird abgebrochen, sobald rho_k unter die Schwelle fällt
    (oder nach max_lag) -- die Faustregel aus Kapitel 4.
    """
    x = np.asarray(x, float)
    if max_lag is None:
        max_lag = int(np.sqrt(len(x)))
    rho = autokorrelation(x, max_lag)
    unter = np.where(rho < schwelle)[0]
    k = unter[0] if len(unter) else max_lag
    return 1.0 + 2.0 * rho[:k].sum()


def se_autokorrelation(x, **kw):
    """Standardfehler des Mittels nach Satz 4.1: s / sqrt(n / tau)."""
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x) / tau_int(x, **kw))


def se_blockmittel(x, bloecke):
    """Standardfehler des Mittels aus Blockmitteln (Satz 4.2)."""
    x = np.asarray(x, float)
    m = len(x) // bloecke
    mittel = x[:m * bloecke].reshape(bloecke, m).mean(axis=1)
    return mittel.std(ddof=1) / np.sqrt(bloecke)


def r_dach(ketten):
    """Skalenreduktionsfaktor von Gelman und Rubin (Definition 8.1).

    ketten: Feld der Form (m Ketten, n Werte).
    """
    C = np.asarray(ketten, float)
    m, n = C.shape
    W = C.var(axis=1, ddof=1).mean()
    B = n * C.mean(axis=1).var(ddof=1)
    V = (n - 1) / n * W + B / n
    return np.sqrt(V / W)


def n_eff_gewichte(w):
    """Wirksame Stichprobengröße (sum w)^2 / sum w^2 (Kapitel 6 und 12)."""
    w = np.asarray(w, float)
    return w.sum() ** 2 / np.dot(w, w)
