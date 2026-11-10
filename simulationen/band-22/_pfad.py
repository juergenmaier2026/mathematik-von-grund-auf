"""Macht das Paket mvg aus dem Nachbarordner importierbar, egal von wo das Skript läuft."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
