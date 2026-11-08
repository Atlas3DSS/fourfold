#!/usr/bin/env python3
"""Summarise think_probe rows: one line per (prompt, arm). python3 think_summary.py rows.jsonl [more.jsonl]"""
import json, re, sys, statistics as st
from collections import defaultdict
SUM_EQ = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
for r in rows:   # older rows: the task-arithmetic metric from the stored answer text (reasoning is not kept in full)
    if "sums" not in r:
        n = wrong = 0
        for lhs, rhs in SUM_EQ.findall((r.get("content") or "") + "\n" + (r.get("reasoning_tail") or "")):
            try:
                v = eval(re.sub(r"[^\d+]", "", lhs)); n += 1; wrong += v != int(rhs.replace(",", ""))  # noqa: S307
            except Exception:
                pass
        r["sums"], r["sums_wrong"] = n, wrong
g = defaultdict(list)
for r in rows: g[(r["prompt"], r["arm"])].append(r)
print(f"{'prompt':18s} {'arm':14s} {'n':>2s} {'tokens med':>10s} {'max':>6s} {'sec med':>8s} {'max':>6s} {'think ch':>9s} {'loops':>5s} {'cut':>3s} {'4 q tags':>8s} {'4 hex ok':>8s} {'sums wrong':>10s} {'eq wrong':>8s} {'wait':>5s}")
for (p, a), rs in sorted(g.items()):
    ok = [r for r in rs if not r["error"]]
    if not ok: print(f"{p:18s} {a:14s} all {len(rs)} errored: {rs[0]['error'][:80]}"); continue
    tok = [r["completion_tokens"] or 0 for r in ok]; sec = [r["seconds"] for r in ok]
    loops = sum(1 for r in ok if r["loop_onset_char"] is not None); cut = sum(1 for r in ok if r["finish"] == "length")
    tags = sum(1 for r in ok if r["answer"]["q_tags"] == 4); ans = sum(1 for r in ok if r["answer"]["q_hex_ok"] == 4)
    eq = sum(r["equations"] for r in ok); wrong = sum(r["equations_wrong"] for r in ok)
    sums = sum(r.get("sums", 0) for r in ok); sums_wrong = sum(r.get("sums_wrong", 0) for r in ok)
    waits = st.mean(r["pivots"]["wait"] for r in ok)
    print(f"{p:18s} {a:14s} {len(ok):2d} {st.median(tok):10.0f} {max(tok):6d} {st.median(sec):8.0f} {max(sec):6.0f} {st.median(r['reasoning_chars'] for r in ok):9.0f} {loops:>3d}/{len(ok):<1d} {cut:3d} {tags:>4d}/{len(ok):<3d} {ans:>4d}/{len(ok):<3d} {sums_wrong:>4d}/{sums:<5d} {wrong:>3d}/{eq:<4d} {waits:5.1f}")
