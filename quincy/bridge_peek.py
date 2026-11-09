#!/usr/bin/env python3
"""Peek at a bridge-format run: per tick action kinds, write types, notes, errors; then one agent's prompt tail and reply.
    python3 quincy/bridge_peek.py RUN_ID [agent] [chars]"""
import json, sys
from collections import Counter
run = sys.argv[1]; agent = int(sys.argv[2]) if len(sys.argv) > 2 else 3; chars = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
rows = [json.loads(l) for l in open(f"runs/{run}/ticks.jsonl")]
ticks = [r for r in rows if r.get("type") == "tick"]
for t in ticks:
    a = t["agents"]
    acts = [k for x in a for k in (x.get("actions") or []) if isinstance(k, dict)]
    kinds = Counter(str(k.get("kind")) for k in acts)
    types = Counter(str(k.get("type")) for k in acts if k.get("kind") in ("say", "ask", "write"))
    errs = Counter(json.dumps(e)[:40] for x in a for e in (x.get("errors") or []))
    pulsed = sum(bool((x.get("usage") or {}).get("pulsed")) for x in a)
    notes = sum(len(x.get("notes") or {}) for x in a)
    nz = sum(1 for x in a for o in x["outputs"] if o.strip("0"))
    print(f"t{t['t']}: pulsed {pulsed}/16 kinds {dict(kinds)} write types {dict(types)} notes {notes} nonzero outs {nz}/64 errors {dict(errs)}")
x = ticks[-1]["agents"][agent]
p = x.get("prompt") or ""; i = p.find("arrived on your input")
print(f"--- tick {ticks[-1]['t']} agent {agent} prompt tail:"); print(p[i:i + 900] if i >= 0 else p[-900:])
print("--- reply:"); print((x.get("raw") or "")[:chars])
