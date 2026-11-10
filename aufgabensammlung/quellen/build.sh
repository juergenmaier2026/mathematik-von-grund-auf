#!/bin/bash
# Vollständiger Bau beider Teilbände inkl. Methodenregister
set -e
cd "$(dirname "$0")"
for b in aufgaben loesungen; do pdflatex -interaction=nonstopmode $b.tex >/dev/null; pdflatex -interaction=nonstopmode $b.tex >/dev/null; done
python3 register.py
for b in aufgaben loesungen; do pdflatex -interaction=nonstopmode $b.tex >/dev/null; pdflatex -interaction=nonstopmode $b.tex >/dev/null; done
for b in aufgaben loesungen; do echo "$b: Fehler $(grep -c '^!' $b.log), $(grep -o 'Output written.*' $b.log)"; done
