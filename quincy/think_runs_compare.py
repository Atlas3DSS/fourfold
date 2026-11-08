#!/usr/bin/env python3
"""Compare fourfold runs tick by tick: wall seconds, reply tokens, how many replies needed the stop pulse, how many agents
had a malformed or missing port value, how many wrote a sum and got it wrong, how many ports carry a non-zero value.
    python3 think_runs_compare.py runs/<id> [runs/<id> ...]"""
import json, re, sys
SUM_EQ = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path.rstrip('/') + '/ticks.jsonl')]
    t = [r for r in rows if r.get('type') == 'tick']
    print(f"== {path.rstrip('/').split('/')[-1]}  ticks={len(t)}")
    print(f"{'tick':>4s} {'wall s':>6s} {'tok med':>7s} {'tok max':>7s} {'pulsed':>6s} {'bad port':>8s} {'no reply':>8s} {'sums':>5s} {'wrong':>5s} {'live ports':>10s}")
    for i, r in enumerate(t):
        ag = r['agents']; tok = [(a.get('usage') or {}).get('completion_tokens') or 0 for a in ag]
        pulsed = sum(1 for a in ag if (a.get('usage') or {}).get('pulsed'))
        bad = sum(1 for a in ag if any(e.startswith('q') and ('malformed' in e or 'missing' in e) for e in (a.get('errors') or [])))
        noreply = sum(1 for a in ag if not (a.get('raw') or '').strip())
        sums = wrong = 0
        for a in ag:
            for lhs, rhs in SUM_EQ.findall((a.get('raw') or '')):
                try:
                    v = eval(re.sub(r"[^\d+]", "", lhs)); sums += 1; wrong += v != int(rhs.replace(',', ''))  # noqa: S307
                except Exception:
                    pass
        live = sum(1 for a in ag for q in (a.get('outputs') or []) if q.strip('0'))
        wall = round(r['time'] - (t[i - 1]['time'] if i else rows[0]['created']))
        print(f"{r['t']:4d} {wall:6d} {sorted(tok)[len(tok)//2]:7d} {max(tok):7d} {pulsed:6d} {bad:8d} {noreply:8d} {sums:5d} {wrong:5d} {live:10d}")
