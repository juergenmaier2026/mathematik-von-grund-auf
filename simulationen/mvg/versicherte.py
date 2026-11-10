"""Der Leitfall von Band 26: Zuzahlung und Arztbesuche.

Jeder Versicherte hat eine Gesundheitsneigung g ~ N(0, 1) (groß = gesund). Wer gesund
ist, wählt eine höhere Zuzahlung; wer gesund ist, geht seltener zum Arzt:
    Zuzahlung   x = 10 + 6 g + eps,            eps ~ N(0, 3^2)
    Besuche     y = max(0, 8 - 0,1 x - 2 g + e), e ~ N(0, 1)
Die wahre Wirkung der Zuzahlung ist -0,1 Besuche je Euro.
"""
import numpy as np

BETA = -0.1


def querschnitt(rng, n, zuzahlung=None):
    """Beobachtungsdaten: g, x, y. Mit `zuzahlung` wird x von außen gesetzt (Experiment, Reform)."""
    g = rng.standard_normal(n)
    x = 10 + 6 * g + rng.normal(0, 3, n) if zuzahlung is None else np.asarray(zuzahlung, float)
    y = np.maximum(0, 8 + BETA * x - 2 * g + rng.standard_normal(n))
    return g, x, y


def experiment(rng, n, stufen=(0, 10, 20, 30)):
    """Zufällige Zuweisung der Zuzahlung (das Krankenversicherungsexperiment)."""
    x = rng.choice(stufen, n).astype(float)
    return querschnitt(rng, n, zuzahlung=x)


def reform(rng, n, erhoehung=7.83):
    """Die Hälfte der Kassen erhöht die Zuzahlung um `erhoehung` -- unabhängig von g.
    Rückgabe: g, x, y, z (Reformkasse ja/nein)."""
    g = rng.standard_normal(n)
    z = rng.random(n) < 0.5
    x = 10 + 6 * g + rng.normal(0, 3, n) + erhoehung * z
    y = np.maximum(0, 8 + BETA * x - 2 * g + rng.standard_normal(n))
    return g, x, y, z.astype(float)


def panel(rng, n, T=6, jahreseffekt=0.1, messfehler=0.0):
    """Dieselben Versicherten über T Jahre: fester Effekt alpha_i = -2 g_i, Jahreseffekte,
    Zuzahlung mit Tarifwechseln und allgemeiner Erhöhung. Rückgabe: x, y als (n, T)."""
    g = rng.standard_normal(n)
    jahre = np.arange(T)
    x = 10 + 6 * g[:, None] + rng.normal(0, 3, (n, T)) + 2.0 * jahre               # Erhöhung um 2 Euro je Jahr
    y = np.maximum(0, 8 + BETA * x - 2 * g[:, None] + jahreseffekt * jahre + rng.standard_normal((n, T)))
    x_beob = x + rng.normal(0, messfehler, (n, T)) if messfehler else x
    return x_beob, y
