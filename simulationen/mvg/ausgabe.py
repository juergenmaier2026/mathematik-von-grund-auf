"""Kleine Hilfen für die Ausgabe der Lehrskripte."""

def kopf(titel):
    """Überschrift einer Tabelle."""
    print()
    print(titel)
    print("-" * len(titel))

def zeile(*spalten, breite=14):
    """Eine Tabellenzeile, alle Spalten rechtsbündig mit zwei Leerzeichen Abstand.

    Gleitkommazahlen mit vier Nachkommastellen; wer anders runden will,
    übergibt einen fertigen Text.
    """
    teile = []
    for s in spalten:
        if isinstance(s, float):
            s = f"{s:.4f}"
        elif isinstance(s, int) and not isinstance(s, bool):
            s = f"{s:,d}".replace(",", " ")
        teile.append(f"{str(s):>{breite}}")
    print("  ".join(teile))

def buch(text):
    """Hinweis auf die Stelle im Buch, deren Zahlen das Skript erzeugt."""
    print(f"   [Buch: {text}]")
