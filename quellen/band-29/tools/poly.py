"""Kleine Polynomarithmetik über Q für Band 29: Gröbner-Basen (Buchberger), Division, Elimination."""
from fractions import Fraction as Fr
from itertools import combinations
class Ring:
    def __init__(self, vars, order='lex'):
        self.vars=list(vars); self.n=len(vars); self.order=order
    def key(self, m):
        if self.order=='lex': return m
        if self.order=='grlex': return (sum(m),)+m
        if self.order=='grevlex': return (sum(m),)+tuple(-e for e in reversed(m))
    def var(self,name):
        i=self.vars.index(name); m=[0]*self.n; m[i]=1
        return Poly(self,{tuple(m):Fr(1)})
    def const(self,c): return Poly(self,{tuple([0]*self.n):Fr(c)} if c!=0 else {})
class Poly:
    def __init__(self,R,terms): self.R=R; self.t={m:c for m,c in terms.items() if c!=0}
    def __add__(self,o):
        o=self._c(o); d=dict(self.t)
        for m,c in o.t.items(): d[m]=d.get(m,0)+c
        return Poly(self.R,d)
    def __neg__(self): return Poly(self.R,{m:-c for m,c in self.t.items()})
    def __sub__(self,o): return self+(-self._c(o))
    def __rsub__(self,o): return self._c(o)-self
    __radd__=__add__
    def __mul__(self,o):
        o=self._c(o); d={}
        for m1,c1 in self.t.items():
            for m2,c2 in o.t.items():
                m=tuple(a+b for a,b in zip(m1,m2)); d[m]=d.get(m,0)+c1*c2
        return Poly(self.R,d)
    __rmul__=__mul__
    def __pow__(self,k):
        r=self.R.const(1)
        for _ in range(k): r=r*self
        return r
    def _c(self,o): return o if isinstance(o,Poly) else self.R.const(o)
    def iszero(self): return not self.t
    def lm(self): return max(self.t, key=self.R.key)
    def lc(self): return self.t[self.lm()]
    def lt(self): m=self.lm(); return Poly(self.R,{m:self.t[m]})
    def monic(self): c=self.lc(); return Poly(self.R,{m:v/c for m,v in self.t.items()})
    def deg(self): return max(sum(m) for m in self.t) if self.t else -1
    def __repr__(self): return self.str()
    def str(self):
        if not self.t: return "0"
        out=[]
        for m in sorted(self.t,key=self.R.key,reverse=True):
            c=self.t[m]; mono="".join((v if e==1 else f"{v}^{e}") for v,e in zip(self.R.vars,m) if e)
            cs = "" if (abs(c)==1 and mono) else str(abs(c))
            s=("-" if c<0 else "+")+cs+mono
            out.append(s)
        s=" ".join(out); return s[1:] if s.startswith("+") else s
    def eval(self,vals):
        return sum(c*prod(Fr(vals[v])**e for v,e in zip(self.R.vars,m)) for m,c in self.t.items())
def prod(it):
    r=Fr(1)
    for x in it: r*=x
    return r
def divides(a,b): return all(x<=y for x,y in zip(a,b))
def mdiv(a,b): return tuple(x-y for x,y in zip(a,b))
def lcm(a,b): return tuple(max(x,y) for x,y in zip(a,b))
def divide(f,G):
    """Mehrfachdivision: f = sum q_i g_i + r. Gibt (quotienten, rest)."""
    R=f.R; q=[R.const(0) for _ in G]; r=R.const(0); p=f
    while not p.iszero():
        m=p.lm(); c=p.lc(); done=False
        for i,g in enumerate(G):
            gm=g.lm()
            if divides(gm,m):
                t=Poly(R,{mdiv(m,gm):c/g.lc()}); q[i]=q[i]+t; p=p-t*g; done=True; break
        if not done:
            r=r+Poly(R,{m:c}); p=p-Poly(R,{m:c})
    return q,r
def spoly(f,g):
    R=f.R; L=lcm(f.lm(),g.lm())
    return Poly(R,{mdiv(L,f.lm()):Fr(1)/f.lc()})*f - Poly(R,{mdiv(L,g.lm()):Fr(1)/g.lc()})*g
def groebner(F):
    G=[f.monic() for f in F if not f.iszero()]
    pairs=[(i,j) for i in range(len(G)) for j in range(i+1,len(G))]
    while pairs:
        i,j=pairs.pop(0)
        # Buchberger-Kriterium: teilerfremde Leitmonome -> überspringen
        if lcm(G[i].lm(),G[j].lm())==tuple(a+b for a,b in zip(G[i].lm(),G[j].lm())): continue
        _,r=divide(spoly(G[i],G[j]),G)
        if not r.iszero():
            G.append(r.monic()); k=len(G)-1
            pairs+=[(i2,k) for i2 in range(k)]
    # reduzieren
    changed=True
    while changed:
        changed=False
        for i in range(len(G)):
            others=G[:i]+G[i+1:]
            if any(divides(h.lm(),G[i].lm()) for h in others):
                G.pop(i); changed=True; break
    H=[]
    for i,g in enumerate(G):
        _,r=divide(g,G[:i]+G[i+1:]); H.append(r.monic())
    return sorted(H,key=lambda p:p.R.key(p.lm()),reverse=True)
def eliminate(G,k):
    """Elementes der (lex-)Gröbner-Basis ohne die ersten k Variablen."""
    return [g for g in G if all(m[i]==0 for m in g.t for i in range(k))]

def hilbert_function(G, dmax):
    """Affine Hilbertfunktion: Zahl der Standardmonome vom Gesamtgrad <= d, d=0..dmax (G: Gröbner-Basis bzgl. gradkompatibler Ordnung)."""
    from itertools import product
    R=G[0].R; n=R.n; lms=[g.lm() for g in G]
    counts=[]
    def mons(d):
        # alle Exponentenvektoren mit Summe == d
        if n==1: yield (d,); return
        for first in range(d+1):
            for rest in mons_n(d-first,n-1): yield (first,)+rest
    def mons_n(d,m):
        if m==1: yield (d,); return
        for first in range(d+1):
            for rest in mons_n(d-first,m-1): yield (first,)+rest
    total=0
    for d in range(dmax+1):
        c=sum(1 for m in mons_n(d,n) if not any(divides(l,m) for l in lms))
        total+=c; counts.append(total)
    return counts
def fit_poly(vals, start):
    """Passt ein Polynom an die Werte vals[start:] an (Differenzen), gibt (Grad, Leitkoeffizient*Grad!)."""
    import math
    seq=vals[start:]; diffs=[seq]; 
    while len(diffs[-1])>1 and any(x!=0 for x in diffs[-1]):
        d=diffs[-1]; diffs.append([d[i+1]-d[i] for i in range(len(d)-1)])
    # Grad = Zahl der Differenzstufen bis konstant
    k=0
    while k<len(diffs) and not all(x==diffs[k][0] for x in diffs[k]): k+=1
    return k, diffs[k][0]
