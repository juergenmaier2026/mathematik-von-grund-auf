"""Teilchenfilter (Band 27, Kapitel 12)."""
import numpy as np


def teilchenfilter(y, log_lik, schritt, start, N, rng, neu_ziehen=True):
    """Bootstrap-Filter (Definition 12.1).

    y          Beobachtungen y_1 .. y_T
    log_lik(y_t, x)   log p(y_t | x) für ein Feld von Teilchen x
    schritt(x, rng)   Vorhersage: Teilchen nach der Zustandsdynamik weiterziehen
    start(rng, N)     Anfangsteilchen
    Rückgabe: Filtermittel je Tag, 5-%- und 95-%-Quantile, wirksame Teilchenzahl je Tag.
    """
    x = start(rng, N)
    logw = np.zeros(N)
    mittel, lo, hi, neff = [], [], [], []
    for yt in y:
        x = schritt(x, rng)                              # Vorhersage
        logw = logw + log_lik(yt, x)                     # Gewichtung
        w = np.exp(logw - logw.max()); w /= w.sum()
        mittel.append(np.dot(w, x))
        idx = np.argsort(x); cw = np.cumsum(w[idx])
        lo.append(x[idx][np.searchsorted(cw, 0.05)]); hi.append(x[idx][np.searchsorted(cw, 0.95)])
        neff.append(1.0 / np.dot(w, w))
        if neu_ziehen:                                   # Neuziehung
            x = x[rng.choice(N, N, p=w)]
            logw = np.zeros(N)
    return np.array(mittel), np.array(lo), np.array(hi), np.array(neff)


def kalman_1d(z, q, r, m0, P0):
    """Kalman-Filter für x_t = x_{t-1} + eps, z_t = x_t + eta (Aufgabe 12.2)."""
    m, P, out = m0, P0, []
    for zt in z:
        P = P + q
        K = P / (P + r)
        m = m + K * (zt - m)
        P = (1 - K) * P
        out.append(m)
    return np.array(out)
