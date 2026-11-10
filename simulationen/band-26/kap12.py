"""Band 26, Kapitel 12 -- Kointegration.

Erzeugt: Superkonsistenz (Satz 12.2) -- der Fehler fällt wie 1/T --, den
Engle-Granger-Test an kointegrierten und an unabhängigen Reihen (12.3) und das
Fehlerkorrekturmodell mit A = ((0,5, 0,3), (0, 0,6)) (Satz 12.4).
"""
import _pfad
import numpy as np
from mvg.oekonometrie import ar1, kq, steigung, dickey_fuller_t
from mvg.ausgabe import kopf, zeile, buch

rng = np.random.default_rng(2612)

kopf("Satz 12.2 -- Superkonsistenz: y = 2 x + u, x Irrfahrt, u stationär; mittlerer |Fehler| aus 2000 Wiederholungen")
zeile("T", "mittl. Fehler", "Fehler * T", breite=16)
for T in (50, 100, 400, 1600):
    f = [abs(steigung(2 * (x := ar1(rng, 1.0, T)) + ar1(rng, 0.5, T), x) - 2) for _ in range(2000)]
    zeile(T, round(float(np.mean(f)), 4), round(float(np.mean(f)) * T, 2), breite=16)
buch("0,0218 / 0,0115 / 0,0031 / 0,0007 -- Fehler mal T bleibt etwa konstant; die Höhe hängt von der Streuung von u ab")

kopf("Engle-Granger: Dickey-Fuller auf den Residuen, kritischer Wert -3,37 (12.3), T = 200")
x = ar1(rng, 1.0, 200); y = 2 * x + ar1(rng, 0.5, 200)
print(f"kointegrierte Reihen: t = {dickey_fuller_t(kq(y, x)[1]):.2f}")
verw = np.mean([dickey_fuller_t(kq(ar1(rng, 1.0, 200), ar1(rng, 1.0, 200))[1]) < -3.37 for _ in range(2000)])
print(f"unabhängige Irrfahrten: verwirft in {verw:.1%} der Fälle")
buch("-7,69 (eine Realisierung) und 4,8 %")

kopf("Satz 12.4 -- Fehlerkorrektur mit A = ((0,5, 0,3), (0, 0,6)): y_2 wirkt auf y_1, nicht umgekehrt; T = 1000")
A = np.array([[0.5, 0.3], [0.0, 0.6]]); T = 1000
Y = np.zeros((T, 2))
for t in range(1, T):
    Y[t] = A @ Y[t - 1] + rng.standard_normal(2)
zeile("Gleichung", "auf y1(t-1)", "auf y2(t-1)", breite=16)
for j in (0, 1):
    b, res, se = kq(Y[1:, j], Y[:-1, 0], Y[:-1, 1])
    zeile(f"y{j + 1}", f"{b[1]:.2f} (t={b[1] / se[1]:.1f})", f"{b[2]:.2f} (t={b[2] / se[2]:.1f})", breite=16)
buch("0,53 (13,6) / 0,31 (7,5); 0,03 (0,9) / 0,59 (14,1)")
