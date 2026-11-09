#!/usr/bin/env python3
"""Summarise a delta_smoke.py jsonl: per shim, actions a reply, hand-written q tags and how many are proper 32-hex, answer length,
finish reasons, sums right/wrong, log tag present; then one sample answer per named shim.
    python3 delta_smoke_report.py smoke2.jsonl [--show d05,d1x48] [--chars 1100]
"""
import argparse, json, re

SUM = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
TAGS = ['say', 'ask', 'compute', 'remember']


def sums(t):
    n = w = 0
    for lhs, rhs in SUM.findall(t):
        try:
            v = eval(re.sub(r"[^\d+]", "", lhs)); n += 1; w += v != int(rhs.replace(",", ""))  # noqa: S307
        except Exception:
            pass
    return n, w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--show', default=''); ap.add_argument('--chars', type=int, default=1100); ap.add_argument('--which', type=int, default=1)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    names = []
    for r in rows:
        if r['shim'] not in names: names.append(r['shim'])
    for name in names:
        rs = [r for r in rows if r['shim'] == name]; ok = [r for r in rs if not r.get('error')]
        if not ok:
            print('==', name, 'errors', [r.get('error') for r in rs][:1]); continue
        c = {t: sum(len(re.findall(rf'<{t}\b', r['answer'])) for r in ok) / len(ok) for t in TAGS}
        qs = sum(len(re.findall(r'<q[0-3]>', r['answer'])) for r in ok) / len(ok)
        goodq = sum(1 for r in ok for m in re.findall(r'<q[0-3]>\s*([0-9a-fA-F]+)\s*</q[0-3]>', r['answer']) if len(m) == 32) / len(ok)
        sm = [sums(r['answer']) for r in ok]
        logs = sum(bool(re.search(r'<log>.*?</log>', r['answer'], re.S)) for r in ok)
        print(f"== {name}: n={len(ok)} actions/reply {sum(c.values()):.2f} (" + ' '.join(f'{k} {v:.2f}' for k, v in c.items()) + f") hand q {qs:.2f} of which 32-hex {goodq:.2f} "
              f"chars {sum(len(r['answer']) for r in ok) // len(ok)} finish {sorted(set(r['finish'] for r in ok))} sums {sum(x[0] for x in sm)} wrong {sum(x[1] for x in sm)} "
              f"log {logs}/{len(ok)} secs {sum(r['secs'] for r in ok) // len(ok)}")
    for name in [x for x in a.show.split(',') if x]:
        ok = [r for r in rows if r['shim'] == name and not r.get('error')]
        if len(ok) <= a.which: continue
        r = ok[a.which]
        print(f"===== {name} tick {r['tick']} agent {r['agent']} finish {r['finish']} chars {len(r['answer'])}")
        print(r['answer'][:a.chars]); print('... TAIL:', r['answer'][-250:])


if __name__ == '__main__':
    main()
