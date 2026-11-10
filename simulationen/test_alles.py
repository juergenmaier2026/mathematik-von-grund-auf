"""Lässt alle Skripte laufen und prüft einige Kennzahlen gegen Formeln.

Aufruf:  python3 test_alles.py
"""
import glob, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvg.erlang import erlang_b
from mvg.station import Intensivstation
from mvg.statistik import se_blockmittel, se_naiv

# 1. Formelprobe: Simulation gegen Erlang B (Stufe 3 der Prüfleiter, Band 27, Satz 13.1)
s = Intensivstation(12)
r = s.lauf(20_000, einschwingen=100, seed=13)
exakt = erlang_b(s.last, 12)
print(f"Erlang-Probe: simuliert {r['anteil']:.4f}, exakt {exakt:.4f}")
assert abs(r["anteil"] - exakt) < 0.01, "Simulation weicht von Erlang B ab"

# 2. Little-Probe
little = s.lam * (1 - r["anteil"]) * s.mittlere_verweildauer
print(f"Little-Probe: Belegung {r['belegung']:.3f}, lambda(1-p)W = {little:.3f}")
assert abs(r["belegung"] - little) < 0.1

# 3. Alle Skripte
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "band-*", "kap*.py"))):
    t = time.time()
    p = subprocess.run([sys.executable, f], capture_output=True, text=True)
    status = "ok" if p.returncode == 0 else "FEHLER"
    print(f"{os.path.relpath(f):28s} {status:7s} {time.time() - t:5.1f} s")
    if p.returncode:
        print(p.stderr)
