#!/usr/bin/env python3
"""Totals per 4x4 arm: ticks, sums written/wrong in replies, malformed-port errors, actions a reply, end eval (sum right, scores).
    python3 quincy/arm_totals.py [run-id ...]   (default: every runs/q-4x4-torus-sum-*-1108 and -1109)"""
import glob, json, re, sys
SUM = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
runs = sys.argv[1:] or sorted(p.split("/")[1] for p in glob.glob("runs/q-4x4-torus-sum-*-110[89]"))
print(f"{'arm':34} ticks  sums wrong  badport  acts/reply  wall/tick | eval: right/16 count wiring total")
for r in runs:
    n = s = w = bad = acts = 0; ticks = 0; wall = []
    prev = None
    for l in open(f"runs/{r}/ticks.jsonl"):
        d = json.loads(l)
        if d.get("type") != "tick": continue
        ticks += 1
        if prev is not None: wall.append(d["time"] - prev)
        prev = d["time"]
        for a in d["agents"]:
            raw = a.get("raw") or ""; n += 1; acts += len(a.get("actions") or [])
            for lhs, rhs in SUM.findall(raw):
                try: v = eval(re.sub(r"[^\d+]", "", lhs)); s += 1; w += v != int(rhs.replace(",", ""))  # noqa: S307
                except Exception: pass
            bad += sum(1 for e in a.get("errors") or [] if "malformed" in e)
    ev = sorted(glob.glob(f"runs/{r}/eval-t*.json")); evs = "no eval yet"
    if ev:
        R = json.load(open(ev[-1]))["results"]; g = lambda k: round(sum(float(x.get(k) or 0) for x in R) / len(R), 3)
        right = sum(1 for x in R if str(x.get("sum_claim")) == str(x.get("sum_true")))
        evs = f"{right}/16 {g('count_score')} {g('wiring_score')} {g('total')}"
    mw = int(sum(wall) / len(wall)) if wall else 0
    print(f"{r.replace('q-4x4-torus-sum-', ''):34} {ticks:5} {s:5} {w:5} {bad:8} {acts / max(1, n):10.2f} {mw:9} | {evs}")
