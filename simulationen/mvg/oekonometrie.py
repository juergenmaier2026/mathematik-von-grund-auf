"""Kleinste Quadrate, Instrumente, Panel- und Zeitreihenhilfen (Band 26)."""
import numpy as np


def kq(y, *X, konstante=True):
    """Kleinste Quadrate: Koeffizienten (Konstante zuerst, wenn konstante=True), Residuen, Standardfehler."""
    A = np.column_stack(X)
    if konstante:
        A = np.column_stack([np.ones(len(y)), A])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    n, k = A.shape
    s2 = res @ res / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return b, res, se


def steigung(y, x):
    """Steigung der einfachen Regression von y auf x."""
    return kq(y, x)[0][1]


def zweistufig(y, x, z):
    """Zweistufige kleinste Quadrate mit einem Regressor x und Instrument z (Satz 3.1 / 3.2).
    Rückgabe: Schätzung und F-Statistik der ersten Stufe."""
    b1, res1, se1 = kq(x, z)
    x_dach = b1[0] + b1[1] * z
    F = (b1[1] / se1[1]) ** 2
    return steigung(y, x_dach), F


def innerhalb(M):
    """Innerhalb-Transformation: von jeder Zeile (Person) ihr Mittel abziehen. M: (Personen, Perioden)."""
    return M - M.mean(axis=1, keepdims=True)


def dickey_fuller_t(y):
    """t-Statistik von gamma in Delta y_t = c + gamma y_{t-1} + e_t."""
    dy, ylag = np.diff(y), y[:-1]
    b, res, se = kq(dy, ylag)
    return b[1] / se[1]


def ar1(rng, phi, T, sigma=1.0, start=0.0):
    """AR(1)-Reihe y_t = phi y_{t-1} + e_t."""
    y = np.empty(T); y[0] = start
    e = rng.normal(0, sigma, T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + e[t]
    return y
