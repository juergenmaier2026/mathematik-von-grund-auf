"""Charaktertafeln und Zerlegungen für Band 30. Gruppen als Permutations-/Matrixgruppen, Klassen, Charaktere."""
import numpy as np, itertools
def closure(gens):
    """Erzeuge die Gruppe aus Matrizen (3x3), als Liste; Gleichheit numerisch."""
    def key(M): return tuple(np.round(M,6).flatten())
    G={key(np.eye(len(gens[0]))):np.eye(len(gens[0]))}
    frontier=list(G.values())
    while frontier:
        new=[]
        for A in frontier:
            for g in gens:
                B=g@A; k=key(B)
                if k not in G: G[k]=B; new.append(B)
        frontier=new
    return list(G.values())
def classes(G):
    def key(M): return tuple(np.round(M,6).flatten())
    inv={key(g):np.linalg.inv(g) for g in G}
    seen=set(); cls=[]
    for g in G:
        if key(g) in seen: continue
        c={}
        for h in G:
            x=h@g@inv[key(h)]; c[key(x)]=x
        for k in c: seen.add(k)
        cls.append(list(c.values()))
    return cls
def decompose(chi_red, table, class_sizes):
    """chi_red: Charakter (Liste über Klassen); table: dict name->Liste; gibt Vielfachheiten."""
    n=sum(class_sizes)
    return {name: round(sum(s*c*np.conj(x) for s,c,x in zip(class_sizes,chi_red,chi))/n,6) for name,chi in table.items()}
