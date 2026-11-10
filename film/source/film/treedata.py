"""Pull the real tree out of a run: for each agent, its parent, the tick it joined the wave, the tick it sent its
subtree's (sum, count) up, and the tick it first held the total. Writes data/<name>.json.

  ../../fourfold/.venv/bin/python film/treedata.py ../../fourfold/runs/<run> data/tree10.json
"""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))   # the fourfold repo root (for sim.py)
import sim

run, out = sys.argv[1], sys.argv[2]
L = [json.loads(l) for l in open(f"{run}/ticks.jsonl")]
meta, ticks = L[0], L[1:]
sim.set_width(meta["config"])
W = meta["agents"]
n = len(W)
res = {"run": meta["run_id"], "rows": meta["rows"], "cols": meta["cols"], "answer": meta["answer"],
       "roots": meta["config"].get("tree_roots") or [0], "ticks": len(ticks), "secrets": [a["secret"] for a in W],
       "pos": [a["pos"] for a in W], "trees": {}}
for x in "AB"[:len(res["roots"])]:
    join, sub, total, parent = {}, {}, {}, {}
    for rec in ticks:
        for a in rec["agents"]:
            i = a["agent"]
            ent = [sim.port_tree(v).get(x) if k == "tree" else None for v, k in zip(a["outputs"], a.get("out_types", []))]
            if i not in join and any(ent):
                join[i] = rec["t"]
            for p, e in enumerate(ent):
                if e and e[0] == "sub" and i not in sub:
                    sub[i] = [rec["t"], e[1], e[2]]
                    parent[i] = W[i]["links"][p]["agent"]
                if e and e[0] == "total" and i not in total:
                    total[i] = [rec["t"], e[1], e[2]]
    res["trees"][x] = {"join": join, "sub": sub, "total": total, "parent": parent}
json.dump(res, open(out, "w"))
t = res["trees"]["A"]
print(res["run"], n, "agents; joined", len(t["join"]), "sub", len(t["sub"]), "total", len(t["total"]),
      "last join", max(t["join"].values()), "last sub", max(v[0] for v in t["sub"].values()), "last total", max(v[0] for v in t["total"].values()))
