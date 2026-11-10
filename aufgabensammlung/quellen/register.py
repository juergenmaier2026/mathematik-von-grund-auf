#!/usr/bin/env python3
"""Erzeugt register_a.tex (Aufgabenband) und register_l.tex (Lösungsband) aus methoden_a.dat und methoden_l.dat."""
import re, unicodedata, collections
def key(s):
    t=unicodedata.normalize('NFKD', s.replace('ß','ss')).encode('ascii','ignore').decode()
    return t.lower()
def nrkey(n):
    a,b=n.split('.'); return (int(a),int(b))
meth=collections.defaultdict(dict)   # method -> {nr: pageA}
for line in open('methoden_a.dat',encoding='utf8'):
    line=line.strip()
    if not line or line.startswith('L|'): continue
    m,nr,pg=[x.strip() for x in line.split('|')]
    meth[m][nr]=pg
lpage={}
try:
    for line in open('methoden_l.dat',encoding='utf8'):
        p=[x.strip() for x in line.strip().split('|')]
        if len(p)==3 and p[0]=='L': lpage[p[1]]=p[2]
except FileNotFoundError: pass
def tex(s): return s.replace('&','\\&').replace('%','\\%')
def build(fn, pages, intro):
    out=[r'\setcounter{gebiet}{0}', r'\chapter*{Methodenregister}', r'\addcontentsline{toc}{chapter}{Methodenregister}', r'\markboth{Methodenregister}{Methodenregister}', intro, r'\begin{multicols}{2}\small\setlength{\parskip}{1pt}', '']
    letter=''
    for m in sorted(meth, key=key):
        L=key(m)[0].upper()
        if L!=letter:
            letter=L; out.append(r'\regbuchstabe{%s}'%L)
        items=[]
        for nr in sorted(meth[m], key=nrkey):
            pg=pages.get(nr) if pages is not lpage else lpage.get(nr)
            items.append('\\mbox{%s (S.\\,%s)}'%(nr, pg) if pg else nr)
        out.append(r'\regeintrag{%s}{%s}'%(tex(m), ', '.join(items)))
    out.append(r'\end{multicols}')
    open(fn,'w',encoding='utf8').write('\n'.join(out)+'\n')
pagesA={nr:pg for m in meth for nr,pg in meth[m].items()}
introA=(r'\worum{Das Register ordnet die Aufgaben nach den Methoden, die sie üben --- nicht nach Gebieten. Wer eine Technik festigen will, findet hier alle Stellen, an denen sie gebraucht wird, quer durch die Bände. Die Methoden stehen auch unter jeder Aufgabe. Angegeben sind Aufgabennummer und Seite in diesem Teilband.}')
introL=(r'\worum{Dasselbe Register wie in Teilband 1, hier mit den Seiten der Lösungen.}')
build('register_a.tex', pagesA, introA)
build('register_l.tex', lpage, introL)
print(len(meth),'Methoden;', sum(len(v) for v in meth.values()),'Einträge')
