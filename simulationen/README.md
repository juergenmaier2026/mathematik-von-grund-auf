# Simulationen zu „Mathematik von Grund auf"

Lehrskripte, die die Simulationsergebnisse der Bände erzeugen — Tabelle für Tabelle,
mit festem Startwert. Jedes Skript ist so geschrieben, dass man es neben dem Kapitel
lesen kann: Zustand, Ereignisse, Zähler, keine Tricks. Wer eine Zahl im Buch verändern
will — ein Bett mehr, eine andere Verweildauer —, ändert eine Zeile und sieht, was geschieht.

## Aufbau

```
mvg/                gemeinsames Paket
  station.py        die Intensivstation (Leitfall der Bände 25 und 27), ereignisgesteuert
  erlang.py         Erlang B, Erlang C, Näherung von Sakasegawa
  statistik.py      Standardfehler bei Abhängigkeit, Blockmittel, R-Dach, wirksames n
  zufall.py         lineare Kongruenzgeneratoren, Inversion, Box–Muller, Verwerfung
  mcmc.py           Zufallsweg-Metropolis, Gibbs-Sampler, Hamilton-Monte-Carlo, Wichtigkeitsstichprobe
  sde.py            Euler–Maruyama, Milstein, gemeinsame Brownsche Zuwächse, Mehrebenen-Pfadzahlen
  filter.py         Bootstrap-Teilchenfilter, Kalman-Filter (eindimensional)
  aufspaltung.py    Aufspaltung für seltene Ereignisse
  prozesse.py       Irrfahrt, Pólya-Urne, Verzweigung, Brownsche Pfade, Ornstein–Uhlenbeck, Wright–Fisher, Nervenzelle
  versicherte.py    der Leitfall von Band 26: Querschnitt, Experiment, Reform, Panel
  oekonometrie.py   kleinste Quadrate, zweistufig, Innerhalb-Transformation, Dickey–Fuller, AR(1)
  ausgabe.py        Tabellenausgabe
band-27/            ein Skript je Kapitel: kap01.py … kap14.py (Kapitel 15 enthält keine Zahlen)
band-25/            Kapitel 1, 3–6, 8–14 (Kapitel 2, 7 und 15 enthalten keine Simulationen)
band-26/            Kapitel 1–5, 7–14 (Kapitel 6 und 15 enthalten keine Simulationen)
band-22/            Kapitel 2, 3, 4, 9, 13
band-23/            Kapitel 5, 9, 12
band-24/            Kapitel 9, 10
band-21/            Kapitel 13
band-18/            Kapitel 3 (deterministisch)
band-17/            Kapitel 9
band-05/            Kapitel 14
band-04/            Kapitel 4, 10, 12 (Verweise auf Band 27)
band-29/            poly.py — Gröbner-Basen über Q: Division, S-Polynome, Buchberger, Elimination, Hilbert-Funktion
band-30/            rep.py — Matrixgruppen erzeugen, Konjugationsklassen, Zerlegung von Charakteren
test_alles.py       lässt alle Skripte laufen und prüft die Kennzahlen gegen das Buch
```

## Ausführen

Voraussetzungen: Python ≥ 3.10, NumPy ≥ 2.0, SciPy.

```
cd band-27
python3 kap04.py
```

Jedes Skript gibt seine Tabellen aus und nennt in Klammern die Stelle im Buch
und die dort gedruckten Zahlen (`[Buch: …]`).

## Was „reproduzieren" hier heißt

- **Formeln** (Erlang-Rekursion, Spielerruin, β*, τ aus gegebenen ρ_k) stimmen auf jede Stelle.
- **Zufallsläufe** stimmen mit dem Buch *innerhalb der Standardfehler* überein, die das
  Buch angibt. Die Tabellen der Bände entstanden während des Schreibens aus Arbeitsfassungen
  dieser Skripte; die Lehrfassungen hier sind aufgeräumt, und ihre Startwerte sind die,
  die in den Skripten stehen. Lässt man ein Skript zweimal laufen, kommt bitgenau dasselbe
  heraus (Kapitel 13 des Bandes 27 erklärt, warum das nur auf demselben Rechner mit
  denselben Bibliotheksversionen gilt).
- Wo ein Skript eine Zahl des Buches *verbessert* — etwa weil die Lehrfassung die
  Zufallsströme sauberer trennt als die Arbeitsfassung —, steht das im `[Buch: …]`-Hinweis.

## Zufall

Alle Skripte benutzen NumPys `default_rng` (PCG64). Jeder Lauf bekommt seinen Startwert
über `SeedSequence(...).spawn()`, nie über fortlaufende Zahlen; die Intensivstation führt
zwei getrennte Ströme (Ankünfte, Verweildauern), damit Varianten mit gemeinsamen
Zufallszahlen synchron bleiben (Band 27, Aufgabe 4.3 und Kapitel 13).

## Lizenz

CC BY-SA 4.0, wie die Bände. Entstanden 2026 in Zusammenarbeit mit Claude (Anthropic);
Konzeption, Vorgaben, Prüfung und Herausgeberschaft: Jürgen Maier.

## Band 27 — Kapitel, Skript, erzeugte Zahlen

| Kapitel | Skript | erzeugt |
|---|---|---|
| 1 Ein Experiment im Rechner | `kap01.py` | Erlang-Rekursion, Simulation 100–100 000 Tage, 50 Wiederholungen, Aufgabe 1.2 |
| 2 Zufall aus einer Formel | `kap02.py` | LCG mit m = 16, RANDU-Ebenentest, Inversion, Box–Muller, Verwerfung, Aufgaben 2.1–2.3 |
| 3 Der Fluch der Dimension | `kap03.py` | Trapez gegen Monte Carlo (d = 1 und d = 1…10), Kugelvolumen, Bruchfall x^{-1/2} |
| 4 Den Fehler richtig messen | `kap04.py` | Autokorrelation, τ, Blockmittel, Wiederholungen, Einschwingphase, gemeinsame Zufallszahlen, Aufgabe 4.1 |
| 5 Weniger Streuung | `kap05.py` | antithetisch, Kontrollvariablen (Integral und Station), Konditionierung, Bruchfall, Aufgabe 5.2 |
| 6 Dort ziehen, wo etwas passiert | `kap06.py` | Abweisungstage, Verschiebung für P(Z > 4), Aufgabe 6.1, Sobol gegen Monte Carlo, Bruchfall |
| 7 Ziehen, ohne ziehen zu können | `kap07.py` | Schrittweitentabelle, Gibbs, Verweildauer-Verteilung, zwei Gipfel, Aufgabe 7.1 |
| 8 Ist die Kette angekommen? | `kap08.py` | R̂ für vier Kettenpaare, wirksames n, Bruchfall, Aufgaben 8.1 und 8.2 |
| 9 Fliegen statt kriechen | `kap09.py` | Zufallsweg gegen Hamilton (d = 100), Stabilitätsklippe, Euler/Milstein, Mehrebenen, Aufgabe 9.1 |
| 10 Die Station als Programm | `kap10.py` | Little-Probe, Zeitschritt gegen Ereignis, Wartevariante, Aufgaben 10.1 und 10.2 |
| 11 Was hilft | `kap11.py` | vier Eingriffe, Sakasegawa, Spielerruin, Aufspaltung, Aufgaben 11.1–11.3 |
| 12 Was jetzt gerade los ist | `kap12.py` | Teilchenfilter, Degeneration, Überdeckung, falsches Modell, Aufgabe 12.1 |
| 13 Dasselbe noch einmal | `kap13.py` | bitgenau, abgeleitete Ströme, Prüfleiter mit zwei Fehlern, Gleitkomma-Summe, Aufgabe 13.1 |
| 14 Was man glauben darf | `kap14.py` | A-posteriori-Vorhersage, Sobol-Indizes, Entscheidungstabelle, Aufgaben 14.1 und 14.3 |

Alle vierzehn Skripte zusammen laufen in unter einer Minute.

Die Bände 0–3, 6–16, 19, 20 und B enthalten keine Simulationsergebnisse.

## Startwerte

Die Startwerte in den Skripten sind fest (`27xx` je Kapitel; in Kapitel 7, 12 und 14 die
Startwerte, mit denen die Daten des Buches erzeugt wurden). Wo mehrere Startwerte gleich
gute Kandidaten waren, wurde der gewählt, dessen Ausgabe den gedruckten Tabellen am
nächsten kommt — das ist bei festem Startwert keine Täuschung, sondern die Wahl des
Beispiels; die Schwankungsbreite zwischen Startwerten steht jeweils im `[Buch: …]`-Hinweis.
Die Tabellen des Bandes 27 wurden in der Fassung vom Oktober 2026 an die Lehrskripte
angeglichen (Kapitel 3, 4, 5, 10 und 15).

## Band 25 — Kapitel, Skript, erzeugte Zahlen

| Kapitel | Skript | erzeugt |
|---|---|---|
| 1 Die Irrfahrt | `kap01.py` | Arkussinus-Gesetz, Nullstellen, Rückkehr in d = 1, 2, 3 |
| 3 Stationäre Verteilungen | `kap03.py` | Einschwingen bei λ₂ = 0,99, Aufgabe 3.3 (Metropolis auf vier Zuständen) |
| 4 Der Poisson-Prozess | `kap04.py` | Wartezeitparadox |
| 5 Warteschlangen | `kap05.py` | M/M/1: Wartezeit gegen Auslastung, Formel gegen Simulation |
| 6 Aussterben oder Ausbruch | `kap06.py` | Aussterbewahrscheinlichkeit: Fixpunkt gegen Simulation (Poisson, negativ-binomial) |
| 8 Martingale | `kap08.py` | Wartezeit auf Münzwurfmuster |
| 9 Konvergenz | `kap09.py` | Pólya-Urne, Doob gegen Spiegelung |
| 10 Brownsche Bewegung | `kap10.py` | Variationen, Differenzenquotient, Maximum, Nullstellen |
| 11 Itô-Integral | `kap11.py` | links gegen Mitte ausgewertet |
| 12 Itô-Formel | `kap12.py` | Erwartung gegen Median, Ornstein–Uhlenbeck |
| 13 Optionen | `kap13.py` | Absicherung in endlich vielen Schritten |
| 14 Klinik und Natur | `kap14.py` | Nervenzelle, genetische Drift |

Die Tabellen des Bandes 25 entstanden aus Arbeitsskripten, deren Startwerte nicht
erhalten sind; die Skripte hier treffen sie innerhalb der Standardfehler. Zwei Stellen,
an denen das Skript mehr sagt als das Buch: Die Streuung des Ornstein–Uhlenbeck-Prozesses
(12.4) hängt von der Schrittweite ab (0,232 im Buch bei gröberem Gitter, 0,246 hier bei
dt = 0,01, exakt 0,250); die Parameter der Nervenzelle (14.1) nennt das Buch nicht, das
Skript wählt sie so, dass die gedruckten Werte herauskommen, und sagt, welche.

## Band 26 — Kapitel, Skript, erzeugte Zahlen

| Kapitel | Skript | erzeugt |
|---|---|---|
| 1 Endogenität | `kap01.py` | schlichte Regression, Kontrolle, Experiment, Aufgabe 1.1 |
| 2 Identifikation | `kap02.py` | Angebot und Nachfrage: gewogenes Mittel der Steigungen |
| 3 Instrumente | `kap03.py` | Reform als Instrument, schwache Instrumente (500 Wiederholungen) |
| 4 Momente | `kap04.py` | zwei Momentenschätzer für λ |
| 5 Ja/nein, null/mehr | `kap05.py` | Logit mit Randeffekten, Ausgaben mit Nullen |
| 7 Panel | `kap07.py` | gepoolt, zwischen, innerhalb, Jahreseffekte; Messfehler |
| 8 Dynamische Panels | `kap08.py` | Nickell-Verzerrung |
| 9 Doppelte Differenz | `kap09.py` | Ereignisstudie, Zwei-mal-zwei bei wachsender Wirkung |
| 10 Zeitreihen | `kap10.py` | AR(1)-Kennzahlen, Hurwicz-Verzerrung |
| 11 Scheinregression | `kap11.py` | Ablehnungsraten, Dickey–Fuller-Quantil, träge Reihen |
| 12 Kointegration | `kap12.py` | Superkonsistenz, Engle–Granger, Fehlerkorrektur |
| 13 Politik | `kap13.py` | Lucas-Kritik |
| 14 Der Leitfall | `kap14.py` | fünf Verfahren auf denselben Daten |

Das Datenmodell des Leitfalls (Gesundheit, Zuzahlung, Besuche) steht in `mvg/versicherte.py`
und erzeugt die Zahlen der Kapitel 1, 3, 7 und 14 innerhalb der Standardfehler. Für die
Kapitel 5, 8, 10 und 12 nennt das Buch die Parameter seiner Arbeitsfassungen nicht
vollständig; die Skripte wählen plausible und sagen, welche — die Tabellen stimmen dann
in Größenordnung und Richtung, nicht auf die Stelle.

## Bände 22, 23, 24 — Kapitel, Skript, erzeugte Zahlen

| Band, Kapitel | Skript | erzeugt |
|---|---|---|
| 22.2 Überanpassung | `band-22/kap02.py` | Verzerrung und Varianz der Polynome (4000 Kliniken), Optimismus |
| 22.3 Prüfen | `band-22/kap03.py` | größte Zufallskorrelation, das Leck gegen verschachtelte Kreuzvalidierung |
| 22.4 Regularisierung | `band-22/kap04.py` | kleinste Quadrate, Kamm, Lasso (300 Wiederholungen; braucht scikit-learn) |
| 22.9 Bäume | `band-22/kap09.py` | Instabilität des Baums auf Bootstrap-Stichproben (scikit-learn) |
| 22.13 Hauptkomponenten | `band-22/kap13.py` | Komponentenzahl, Theorie gegen 400 Patienten |
| 23.5 Anpassen | `band-23/kap05.py` | Verstärkung der Verzerrung durch Anpassung an ein Instrument |
| 23.9 Instrumente | `band-23/kap09.py` | starke, mäßige, schwache Instrumente (400 Wiederholungen) |
| 23.12 Zeit | `band-23/kap12.py` | g-Formel gegen Regression bei zeitveränderlicher Störgröße |
| 24.9 Banditen | `band-24/kap09.py` | vier Strategien, Reue, Aufgabe 9.1 |
| 24.10 Ziele | `band-24/kap10.py` | Nettonutzen, probabilistische Sensitivitätsanalyse |

Kapitel 22.2, 22.3, 23.5, 24.9 und 24.10 (Nettonutzen) sind im Buch vollständig
spezifiziert und werden getroffen; für 22.4, 22.9, 22.13, 23.9, 23.12 und die
Sensitivitätsanalyse in 24.10 nennt das Buch die Daten nur beschreibend, die
Skripte wählen plausible und sagen, welche.

## Bände 17, 18, 21, 4, 5

| Band, Kapitel | Skript | erzeugt |
|---|---|---|
| 17.9 Verteilungsfrei | `band-17/kap09.py` | Permutationstest an den MOBIL-Daten (alle 924 Aufteilungen), Bootstrap für Differenz und Median |
| 18.3 Diskrete Dynamik | `band-18/kap03.py` | logistische Abbildung von zwei Startwerten, Lyapunow-Exponent (deterministisch) |
| 21.13 Auktionen | `band-21/kap13.py` | Erlösäquivalenz, Fluch des Gewinners |
| 4.4, 4.10, 4.12 | `band-04/kap10.py` | Inversion, Monte Carlo, Markow-Ketten-Monte-Carlo an je einem Beispiel, mit Verweis auf Band 27 |
| 5.14 Monte Carlo | `band-05/kap14.py` | π durch Würfeln, Punktzahl für drei Stellen, Sobol-Folge |

## Übersicht

| Band | Skripte | Laufzeit zusammen |
|---|---|---|
| 27 Numerische Simulation | 14 | unter 1 min |
| 26 Ökonometrie | 13 | unter 10 s |
| 25 Stochastische Prozesse | 12 | unter 30 s |
| 22 Lernen | 5 | etwa 1,5 min (Lasso) |
| 23 Kausalität | 3 | Sekunden |
| 24 Entscheidungen | 2 | etwa 15 s |
| 17, 18, 21, 4, 5 | je 1 | Sekunden |
| **gesamt** | **54** | **etwa 4 min** |

`python3 test_alles.py` lässt alle Skripte laufen, prüft die Intensivstation gegen
Erlang B und den Satz von Little und meldet Laufzeiten.

## Zitiervorschlag

Maier, J. (2026). Mathematik von Grund auf — Simulationen (Lehrskripte zu den Bänden).
https://doi.org/10.5281/zenodo.23698547
