"""Scripted (no-LLM) model: exact network sum on a torus with fourfold's channel limits.

Each agent has 4 ports (N/E/S/W, torus wrap); a port carries at most K=2 id:number pairs per tick
(fourfold: 32 hex chars, 16 per pair). All agents are perfect and synchronous.

flood: every agent keeps the set of ids it knows and, per port, sends the K oldest pairs it has not
       yet sent on that port. Done when every agent knows all N ids. (The best case for flooding.)
tree:  echo/convergecast (Chang 1982; Segall 1983). Root 0 starts a wave; a node's parent is the
       port the wave first arrived on; children are learned from the wave reply; each node sends
       (sum, count) of its subtree to its parent once all children reported; root then floods the
       total back down. One (sum, count) pair per message.
"""
import sys
from collections import deque

K = 2


def nbrs(R, C):
    out = []
    for i in range(R * C):
        r, c = divmod(i, C)
        out.append([((r - 1) % R) * C + c, r * C + (c + 1) % C, ((r + 1) % R) * C + c, r * C + (c - 1) % C])
    return out


def flood(R, C, cap=100000):
    N = R * C
    nb = nbrs(R, C)
    known = [{i} for i in range(N)]
    order = [[i] for i in range(N)]                      # arrival order of ids per agent
    sent = [[0] * 4 for _ in range(N)]                    # per port: index into order already sent
    t = 0
    while t < cap:
        if all(len(k) == N for k in known):
            return t
        t += 1
        inbox = [[] for _ in range(N)]
        for a in range(N):
            for p in range(4):
                s = sent[a][p]
                batch = order[a][s:s + K]
                sent[a][p] = s + len(batch)
                inbox[nb[a][p]].extend(batch)
        for a in range(N):
            for x in inbox[a]:
                if x not in known[a]:
                    known[a].add(x); order[a].append(x)
    return None


def tree(R, C):
    N = R * C
    nb = nbrs(R, C)
    dist = [-1] * N; dist[0] = 0; q = deque([0])
    while q:
        u = q.popleft()
        for v in nb[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; q.append(v)
    D = max(dist)
    # wave reaches depth d at tick d; a node learns its children 2 ticks after its own wave (forward + reply);
    # deepest leaves report at D+1, each level up adds one tick (children all report by then): root has the
    # total at about 2D+2; broadcasting it down adds D. Upper bound, perfect agents.
    return D, 2 * D + 2, 3 * D + 2


if __name__ == "__main__":
    shapes = [(4, 4), (10, 10), (4, 28), (20, 20), (32, 32), (64, 64)]
    print(f"{'world':>8} {'agents':>6} {'diam':>4} {'flood ticks':>11} {'floor N/8':>9} {'tree: root has sum':>18} {'tree: all have sum':>18}")
    for R, C in shapes:
        N = R * C
        f = flood(R, C) if N <= 1024 else None
        D, up, all_ = tree(R, C)
        print(f"{R}x{C:<5} {N:>6} {D:>4} {str(f) if f is not None else 'skipped':>11} {N/8:>9.0f} {up:>18} {all_:>18}")
        sys.stdout.flush()
