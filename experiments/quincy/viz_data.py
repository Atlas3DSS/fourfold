"""compact per-tick data for the torus viewer, from a run's ticks.jsonl(.gz) and eval json.

  python experiments/quincy/viz_data.py RUN_DIR > run.json
  python experiments/quincy/viz_data.py RUN_DIR --source 37 > run.json   # also trace influence from agent 37

per agent per tick, one state digit (the highest that applies):
  0 quiet       outputs unchanged this tick
  1 active      outputs changed
  2 social      its notes talk about other processes / not being alone
  3 neighbour   its notes contain a true neighbour's id
  4 text        an output carries readable ascii (4+ letters in a row)
plus an error flag, the first log line, and which links changed value this tick.

--source K adds an influence trace (strict origin taint): an agent is reached at the first tick an
input carries an 8-hex chunk that first appeared on the network at agent K, or at an agent K had
already reached after it was reached. relayed values from elsewhere never count.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

SOCIAL = re.compile(r"(another|other|neighbo\w*|peer|external|remote) (process|agent|entity|node|sender)"
                    r"|not alone|someone|communicat\w*|handshake|\bmessage")
WORD = re.compile(rb"[A-Za-z]{4,}")


def load(run: Path):
    f = run / "ticks.jsonl"
    if f.exists():
        fh = open(f, encoding="utf-8")
    else:
        fh = gzip.open(run / "ticks.jsonl.gz", "rt", encoding="utf-8")
    with fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    meta = next(l for l in lines if l["type"] == "meta")
    ticks = [l for l in lines if l["type"] == "tick"]
    evals = sorted(run.glob("eval-t*.json"), key=lambda p: int(p.stem.split("-t")[1]))
    return meta, ticks, (json.loads(evals[-1].read_text()) if evals else None)


def chunks(v: str) -> set[str]:
    """nonzero, non-repetitive 8-hex chunks of a port value, used as fingerprints."""
    out = set()
    for i in range(0, len(v) - 7, 8):
        c = v[i:i + 8]
        if c.strip("0") and len(set(c)) > 2:
            out.add(c)
    return out


def build(run: Path, source: int | None = None) -> dict:
    meta, ticks, ev = load(run)
    agents = meta["agents"]
    n = len(agents)
    seeds = [a["seed"] for a in agents]
    nb = [{l["agent"] for l in a["links"] if l} for a in agents]
    # directed links (a, port) -> (b, port); keep each undirected pair once for drawing
    pairs, idx = [], {}
    for a in agents:
        for i, l in enumerate(a["links"]):
            if l:
                key = tuple(sorted(((a["id"], i), (l["agent"], l["port"]))))
                if key not in idx:
                    idx[key] = len(pairs)
                    pairs.append([key[0][0], key[1][0], 1 if l.get("seam") else 0])
    frames, prev = [], None
    reached = {source: 0} if source is not None else {}
    emitted: dict[str, int] = {}          # traced fingerprint -> tick it first appeared
    origin: dict[str, tuple] = {}         # every fingerprint -> (tick, agent) of its first appearance
    for t in ticks:
        A = t["agents"]
        state, err, logs = [], [], []
        changed_links = [0] * len(pairs)
        for i, a in enumerate(A):
            notes = (a["protocol"] + " " + " ".join(a["log"])).lower()
            s = 0
            if prev is not None and a["outputs"] != prev[i]["outputs"]:
                s = 1
            if SOCIAL.search(notes):
                s = 2
            if any(seeds[j] in notes for j in nb[i]):
                s = 3
            if any(WORD.search(bytes.fromhex(v)) for v in a["outputs"]):
                s = 4
            state.append(s)
            err.append(1 if a["errors"] else 0)
            logs.append((a["log"][0] if a["log"] else "")[:140])
            if prev is not None:
                for p, l in enumerate(agents[i]["links"]):
                    if l and a["outputs"][p] != prev[i]["outputs"][p]:
                        key = tuple(sorted(((i, p), (l["agent"], l["port"]))))
                        changed_links[idx[key]] = 1
        if source is not None:
            # strict origin taint: a chunk is traced only if it first appeared anywhere on the network
            # at the source, or at an already-reached agent after it was reached. relaying someone
            # else's value therefore never credits the source.
            for i, a in enumerate(A):
                if i not in reached and any(c in emitted and emitted[c] < t["t"]
                                            for v in a["inputs"] for c in chunks(v)):
                    reached[i] = t["t"]
            for i, a in enumerate(A):
                for v in a["outputs"]:
                    for c in chunks(v):
                        if c not in origin:
                            origin[c] = (t["t"], i)
                            if i in reached and t["t"] >= reached[i]:
                                emitted[c] = t["t"]
        frames.append({"t": t["t"], "s": "".join(map(str, state)), "e": "".join(map(str, err)),
                       "l": "".join(map(str, changed_links)), "log": logs})
        prev = A
    out = {
        "run": run.name, "rows": meta["rows"], "cols": meta["cols"], "topology": meta.get("topology", "grid"),
        "prime": meta["config"].get("prime", ""), "model": meta["config"].get("llm", {}).get("model", ""),
        "task": meta["config"].get("task") or "", "actions": bool(meta["config"].get("actions")),
        "modules": meta["config"].get("modules"), "seam_period": meta["config"].get("seam_period") or 1,
        "answer": meta.get("answer"),
        "agents": [{"seed": a["seed"], "pos": a["pos"]} for a in agents],
        "links": pairs, "frames": frames,
        "protocols": [a["protocol"][:1600] for a in ticks[-1]["agents"]],
    }
    if source is not None:
        out["source"] = source
        out["reached"] = {str(k): v for k, v in reached.items()}
    if ev:
        out["eval"] = {"tick": ev["tick"], "count": [r["count"] for r in ev["results"]],
                       "total": [r["total"] for r in ev["results"]],
                       "arrangement": [(r["arrangement"] or "")[:160] for r in ev["results"]]}
    return out


if __name__ == "__main__":
    src = int(sys.argv[sys.argv.index("--source") + 1]) if "--source" in sys.argv else None
    print(json.dumps(build(Path(sys.argv[1]), src), separators=(",", ":"), ensure_ascii=False))
