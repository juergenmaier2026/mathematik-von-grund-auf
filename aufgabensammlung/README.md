# Aufgabensammlung zu „Mathematik von Grund auf" — Arbeitsfassung

Zwei Teilbände: **Teil 1** enthält die Aufgaben, **Teil 2** die vollständigen Lösungen.
Die Sammlung ist nach Gebieten geordnet, die den Bänden der Reihe folgen; jedes Gebiet
enthält drei Sorten von Aufgaben:

- **Typische Aufgaben** — das, was man in diesem Gebiet rechnen können muss,
- **Stolpersteine** — Aufgaben zu den häufigen Fehlern, jeweils mit einem Kasten „Der typische Fehler",
- **Aus dem Werk** — die Übungsaufgaben der Bände, hier noch einmal mit ausführlicher Lösung.

Jede Aufgabe hat einen Schwierigkeitsgrad (1–3), eine Zeitangabe und am Rand einen
*Stupser* (einen Hinweis, der die Richtung zeigt, ohne die Lösung zu verraten); jede
Lösung beginnt mit einem zweiten Stupser und endet mit einer Probe und einem Kasten
„Mitnehmen". Ein Methodenregister am Ende beider Teile führt zu allen Aufgaben, in
denen eine Methode vorkommt.

## Stand

**Arbeitsfassung, Gebiete 0–9** (Schulmathematik, Zahlentheorie, lineare Algebra, Analysis,
Wahrscheinlichkeit, Numerik, Geometrie, Mathematikgeschichte, Topologie,
Differentialgleichungen) — 420 Aufgaben. Die weiteren Gebiete folgen in der
Reihenfolge der Bände; mit der vollständigen Sammlung wird die Fassung 1.2 der Reihe
erscheinen. Die fertigen Gebiete sind benutzbar; Nummern und Seitenzahlen der
fertigen Gebiete ändern sich nicht mehr.

## Bauen

```
cd quellen && ./build.sh
```

(TeX Live mit Palatino, TikZ, tcolorbox, marginnote; Python 3 für das Methodenregister.)
Erzeugt `aufgaben.pdf` und `loesungen.pdf`.

## Lizenz

Wie die Reihe: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.de).
