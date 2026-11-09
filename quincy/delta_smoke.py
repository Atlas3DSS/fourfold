#!/usr/bin/env python3
"""Smoke test for the answer lace: send sampled flare-trigger prompts through one or more shims and count what the answers do.

    python3 delta_smoke.py --samples delta-1108/samples.json --run q-4x4-torus-sum-flare-trigger-1108 --n 6 \
        --shim plain=http://127.0.0.1:9436/v1 --shim delta1=http://127.0.0.1:9440/v1 [--seeds 1] [--concurrency 4]
Prints per shim: actions a reply (say/ask/compute/remember), hand-written q tags, answer chars, pulsed share, sums right/wrong,
and the first answer in full for a look. Same prompts and seeds for every shim.
"""
import argparse, concurrent.futures, json, re, time, urllib.request

SUM_EQ = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
TAGS = ['say', 'ask', 'compute', 'remember', 'recall', 'move', 'protocol', 'log']


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sums(text):
    n = wrong = 0
    for lhs, rhs in SUM_EQ.findall(text):
        try:
            v = eval(re.sub(r"[^\d+]", "", lhs)); n += 1; wrong += v != int(rhs.replace(",", ""))  # noqa: S307
        except Exception:
            pass
    return n, wrong


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', required=True); ap.add_argument('--run', required=True); ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--shim', action='append', required=True, help='name=url'); ap.add_argument('--seeds', type=int, default=1); ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--effort', default='xhigh'); ap.add_argument('--out', default='')
    ap.add_argument('--min-tick', type=int, default=0, help='use prompts from this tick on (late ticks carry the history that makes pulsed replies passive)')
    a = ap.parse_args()
    samples = [s for s in json.load(open(a.samples)) if s['run'] == a.run and s['tick'] >= a.min_tick][:a.n]
    shims = dict(x.split('=', 1) for x in a.shim)
    jobs = [(name, url, s, seed) for name, url in shims.items() for s in samples for seed in range(a.seeds)]

    def one(job):
        name, url, s, seed = job
        body = dict(model='qwen38-27b', messages=[dict(role='system', content=s['system']), dict(role='user', content=s['user'])], max_tokens=8192, seed=1000 + seed,
                    reasoning_effort=a.effort)
        t0 = time.time()
        try:
            r = post(url + '/chat/completions', body)
            m = r['choices'][0]['message']; u = r.get('usage') or {}
            return dict(shim=name, tick=s['tick'], agent=s['agent'], seed=seed, secs=round(time.time() - t0), answer=m.get('content') or '', reasoning_chars=len(m.get('reasoning_content') or ''),
                        pulsed=bool(u.get('pulsed')), laced=bool(u.get('answer_laced')), finish=r['choices'][0].get('finish_reason'))
        except Exception as e:  # noqa
            return dict(shim=name, tick=s['tick'], agent=s['agent'], seed=seed, error=repr(e)[:300], answer='')

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(one, jobs))
    if a.out:
        with open(a.out, 'w') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
    for name in shims:
        rs = [r for r in rows if r['shim'] == name]; ok = [r for r in rs if not r.get('error')]
        if not ok:
            print(f'== {name}: all failed: {rs[0].get("error")}'); continue
        counts = {t: sum(len(re.findall(rf'<{t}\b', r['answer'])) for r in ok) / len(ok) for t in TAGS}
        qs = sum(len(re.findall(r'<q[0-3]>', r['answer'])) for r in ok) / len(ok)
        acts = counts['say'] + counts['ask'] + counts['compute'] + counts['remember'] + counts['recall']
        sm = [sums(r['answer']) for r in ok]
        print(f"== {name}: {len(ok)}/{len(rs)} replies, pulsed {sum(r['pulsed'] for r in ok)}, laced {sum(r['laced'] for r in ok)}, "
              f"actions/reply {acts:.2f} (say {counts['say']:.2f} ask {counts['ask']:.2f} compute {counts['compute']:.2f} remember {counts['remember']:.2f}), "
              f"hand q tags {qs:.2f}, answer chars {sum(len(r['answer']) for r in ok) // len(ok)}, sums {sum(x[0] for x in sm)} wrong {sum(x[1] for x in sm)}, "
              f"finish {sorted(set(r['finish'] for r in ok))}, secs {sum(r['secs'] for r in ok) // len(ok)}")
        print('--- first answer:'); print(ok[0]['answer'][:1500]); print()


if __name__ == '__main__':
    main()
