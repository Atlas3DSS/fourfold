import sys; sys.path.insert(0, '.')
import flood_vs_tree as m
def tess_nb(n):
    idx = lambda c: ((c[0]*n + c[1])*n + c[2])*n + c[3]
    out = []
    for a in range(n**4):
        c = [a//n**3 % n, a//n**2 % n, a//n % n, a % n]
        row = []
        for i in range(4):
            d = list(c); d[i] = (d[i]+1) % n; row.append(idx(d))   # directed: port i sends +1 in dim i
        out.append(row)
    return out
m.nbrs = lambda R, C: tess_nb(int(round(R ** 0.5)))
for n in (3, 4, 5):
    N = n**4
    f = m.flood(n*n, n*n)
    print(f"tesseract side {n}: {N} agents, directed diameter {4*(n-1)}, perfect flood {f} ticks, per-dimension ring sum {4*(n-1)} ticks")
