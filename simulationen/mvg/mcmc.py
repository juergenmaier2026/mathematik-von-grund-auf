"""Markow-Ketten-Verfahren (Band 27, Kapitel 7 bis 9)."""
import numpy as np


def metropolis_zufallsweg(log_dichte, start, schritt, n, rng):
    """Zufallsweg-Metropolis (Definition 7.1 mit symmetrischem Vorschlag).

    log_dichte(x)  Logarithmus der Zieldichte bis auf eine Konstante
    start          Startpunkt (Vektor)
    schritt        Streuung des Vorschlags x' = x + schritt * Z
    Rückgabe: Kette (n, d) und Annahmerate.
    """
    x = np.array(start, float)
    lp = log_dichte(x)
    kette = np.empty((n, len(x)))
    angenommen = 0
    for i in range(n):
        vorschlag = x + schritt * rng.standard_normal(len(x))
        lp_neu = log_dichte(vorschlag)
        if np.log(rng.random()) < lp_neu - lp:        # bergauf immer, bergab mit Wahrscheinlichkeit e^(Differenz)
            x, lp = vorschlag, lp_neu
            angenommen += 1
        kette[i] = x                                 # abgelehnt: der alte Wert zählt noch einmal
    return kette, angenommen / n


def gibbs_normal(y, n, rng, start=None):
    """Gibbs-Sampler für (mu, sigma) einer Normalverteilung bei flacher A-priori-Dichte
    in mu und log sigma (Kapitel 7.4):
        mu | sigma ~ N(ybar, sigma^2/n_y),   sigma^2 | mu ~ sum (y-mu)^2 / chi^2_{n_y}
    Rückgabe: Kette (n, 2) mit Spalten mu, sigma.
    """
    y = np.asarray(y, float)
    ny, ybar = len(y), y.mean()
    mu, sigma = (ybar, y.std(ddof=1)) if start is None else start
    kette = np.empty((n, 2))
    for i in range(n):
        mu = rng.normal(ybar, sigma / np.sqrt(ny))
        sigma = np.sqrt(np.sum((y - mu) ** 2) / rng.chisquare(ny))
        kette[i] = mu, sigma
    return kette


def hamilton(U, grad_U, start, eps, L, n, rng):
    """Hamilton-Monte-Carlo (Definition 9.1) mit dem Sprungverfahren.

    U(x)        potentielle Energie = -log Zieldichte
    grad_U(x)   ihr Gradient
    eps, L      Schrittweite und Zahl der Sprungschritte
    Rückgabe: Kette (n, d), Annahmerate, Liste der Energiefehler Delta H.
    """
    x = np.array(start, float)
    d = len(x)
    kette = np.empty((n, d))
    angenommen = 0
    dH = np.empty(n)
    for i in range(n):
        p = rng.standard_normal(d)
        x0, H0 = x.copy(), U(x) + 0.5 * p @ p
        p = p - 0.5 * eps * grad_U(x)                # halber Impulsschritt
        for l in range(L):
            x = x + eps * p                          # Ortsschritt
            if l < L - 1:
                p = p - eps * grad_U(x)              # voller Impulsschritt
        p = p - 0.5 * eps * grad_U(x)                # halber Impulsschritt
        H1 = U(x) + 0.5 * p @ p
        dH[i] = H1 - H0
        if np.log(rng.random()) < H0 - H1:
            angenommen += 1
        else:
            x = x0
        kette[i] = x
    return kette, angenommen / n, dH


def wichtigkeit(h, log_f, log_g, ziehe_g, n, rng):
    """Wichtigkeitsstichprobe (Definition 6.1): Schätzung, Standardfehler, wirksames n."""
    x = ziehe_g(rng, n)
    w = np.exp(log_f(x) - log_g(x))
    y = h(x) * w
    # wirksames n der Beiträge |h| w -- für ein seltenes Ereignis zählen nur die Treffer
    v = np.abs(y)
    n_eff = v.sum() ** 2 / np.dot(v, v) if v.any() else 0.0
    return y.mean(), y.std(ddof=1) / np.sqrt(n), n_eff
