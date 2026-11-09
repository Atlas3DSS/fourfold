#!/usr/bin/env python3
"""Act-vs-passive delta between two fourfold runs (fabe-integration, 2026-11-08).

Owner's question: does the stop pulse make the agents not want to act? Flare-trigger replies (xhigh + pulse at 2,048) use
0.54 actions a reply against 2.66 at medium, with answers the same length. This tool measures the model's residual state
while it writes those answers, in both runs, and takes the difference: delta_L = mean_medium_L - mean_flare_L per layer.
The delta is saved in the think.npz layout (layers, d, n_w) so flare_pulse_shim2.py can add it back during the answer.

Two stages, two interpreters (sim needs httpx, the engine client needs transformers):
  stage 1, fourfold's .venv:   python delta_capture.py prompts --runs RUN_MED RUN_FLARE --out DIR [--ticks 2-9] [--per-tick 3]
      -> DIR/samples.json : for each sampled reply, the exact chat messages, the reasoning, the answer, the effort
  stage 2, the flare venv:     python delta_capture.py capture --engine http://127.0.0.1:893x --out DIR [--fracs 0.1,0.3,0.5,0.7,0.9]
      -> DIR/reads.npz (per read: run, tick, agent, pos, residual (64, 5120) float16), DIR/delta.npz, DIR/report.txt
The engine is neuropedia's batch thinking server (think_edit_board.sh): max_tokens 1 with think.prefix_token_ids gives the
block-output residual of all 64 layers at the last position (residual_last). Positions: 'close' = the think so far +
</think>, and the answer at the given fractions of its length.
"""
import argparse, json, os, random, sys, time, urllib.request, uuid
from pathlib import Path

END_THINK = 248069


def say(**row):
    row['at'] = round(time.time(), 1); print(json.dumps(row, default=str), flush=True)


# ---------------------------------------------------------------- stage 1: the prompts, from the runs
def prompts(a):
    sys.path.insert(0, str(Path(a.fourfold)))
    import sim
    samples = []
    lo, hi = (int(x) for x in a.ticks.split('-'))
    for run in a.runs:
        path = Path(a.fourfold) / 'runs' / run
        meta, ticks = sim.load_run(path / 'ticks.jsonl')
        effort = ((meta['config'].get('llm') or {}).get('extra_body') or {}).get('reasoning_effort') or 'medium'
        say(event='run', run=run, ticks=len(ticks), effort=effort, prime=meta['config'].get('prime'), task=meta['config'].get('task'),
            seed=meta.get('world_seed'), agents=len(meta['agents']))
        rng = random.Random(f'delta:{run}')
        for rec in ticks:
            t = rec['t']
            if not lo <= t <= hi:
                continue
            ids = list(range(len(rec['agents']))); rng.shuffle(ids)
            for i in ids[:a.per_tick]:
                r = rec['agents'][i]
                if not r.get('raw') or not r.get('prompt'):
                    continue
                samples.append(dict(run=run, tick=t, agent=r['agent'], effort=effort, pulsed=bool((r.get('usage') or {}).get('pulsed')),
                                    system=sim.system_prompt(meta, meta['agents'][r['agent']]['seed']), user=r['prompt'],
                                    reasoning=r.get('reasoning') or '', answer=r['raw']))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(out / 'samples.json', 'w'), ensure_ascii=False)
    say(event='samples', n=len(samples), per_run={r: sum(s['run'] == r for s in samples) for r in a.runs},
        no_reasoning=sum(not s['reasoning'] for s in samples), path=str(out / 'samples.json'))


# ---------------------------------------------------------------- stage 2: the reads
def http(url, payload=None, timeout=7200):
    data = None if payload is None else json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} from {url}: {e.read().decode("utf-8", "replace")[:600]}') from None


class Engine:
    def __init__(self, url):
        import numpy as np
        self.np = np; self.url = url; d = http(url + '/v1/models', timeout=60)['data'][0]
        self.model_id = d['id']; b = d['flare']['batch']; self.page_tokens = b['page_tokens']; self.pages = b['physical_pages']; self.max_batch = b.get('max_batch_size', 4)
        say(event='engine', model_id=self.model_id, pages=self.pages, page_tokens=self.page_tokens)

    def batch(self, reqs, timeout=7200):
        np = self.np
        for r in reqs:
            r.setdefault('model', self.model_id); r.setdefault('stream', False); r.setdefault('metadata', {'job_id': 'delta-' + uuid.uuid4().hex[:10]})
        batch = {'batch_id': 'delta-' + uuid.uuid4().hex[:12], 'requests': reqs}
        pre = http(self.url + '/v1/batch/preflight', batch, timeout=300)
        if pre.get('fits') is not True:
            raise RuntimeError('preflight: ' + str(pre.get('reason')))
        t0 = time.time(); raw = http(self.url + '/v1/batch/completions', batch, timeout=timeout); secs = time.time() - t0
        out = []
        for item in raw['results']:
            resp = item['response']; art = resp['activation_artifact']['path']
            with np.load(art, allow_pickle=False) as z:
                R = z['residual_last'].astype(np.float32); ids = z['generated_token_ids'].tolist(); plen = len(z['prompt_token_ids'])
            for key in ('activation_artifact', 'provenance_artifact'):
                p = (resp.get(key) or {}).get('path')
                if p and p.startswith('/mnt/storage16tb/fourfold/'):
                    try: os.remove(p)
                    except OSError: pass
            out.append(dict(R=R, next=ids[0] if ids else None, plen=plen, secs=secs))
        return out


def capture(a):
    import numpy as np
    from transformers import AutoTokenizer
    out = Path(a.out); samples = json.load(open(out / 'samples.json'))
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    e = Engine(a.engine)
    fracs = [float(x) for x in a.fracs.split(',')]
    done = {}
    reads_path = out / 'reads.npz'
    if reads_path.exists():
        z = np.load(reads_path, allow_pickle=False)
        for k in range(len(z['run'])):
            done[(str(z['run'][k]), int(z['tick'][k]), int(z['agent'][k]), str(z['pos'][k]))] = (z['R'][k], int(z['next'][k]))
        say(event='resume', reads=len(done))
    jobs = []
    for s in samples:
        think_ids = tok(s['reasoning'], add_special_tokens=False)['input_ids'] if s['reasoning'] else []
        ans_ids = tok('\n\n' + s['answer'], add_special_tokens=False)['input_ids']
        n_prompt = len(tok.apply_chat_template([dict(role='system', content=s['system']), dict(role='user', content=s['user'])], tokenize=True,
                                               add_generation_prompt=True, return_dict=False, enable_thinking=True, reasoning_effort=s['effort']))
        positions = [('close', think_ids + [END_THINK])] + [(f'a{f:.2f}', think_ids + [END_THINK] + ans_ids[:max(1, int(len(ans_ids) * f))]) for f in fracs]
        for pos, prefix in positions:
            key = (s['run'], s['tick'], s['agent'], pos)
            if key in done:
                continue
            jobs.append(dict(key=key, s=s, prefix=prefix, pages=-(-(n_prompt + len(prefix) + 2) // e.page_tokens)))
    say(event='plan', reads=len(jobs), samples=len(samples), fracs=fracs, mean_pages=round(sum(j['pages'] for j in jobs) / max(1, len(jobs)), 1))
    jobs.sort(key=lambda j: (j['key'][0], j['key'][1], j['key'][2]))

    def flush():
        keys = sorted(done)
        np.savez(reads_path, run=np.array([k[0] for k in keys]), tick=np.array([k[1] for k in keys]), agent=np.array([k[2] for k in keys]),
                 pos=np.array([k[3] for k in keys]), R=np.stack([done[k][0] for k in keys]).astype(np.float16), next=np.array([done[k][1] for k in keys]))

    i = 0; t_start = time.time(); n0 = len(done)
    while i < len(jobs):
        part, used = [], 0
        while i < len(jobs) and len(part) < e.max_batch and used + jobs[i]['pages'] <= e.pages - 2:
            part.append(jobs[i]); used += jobs[i]['pages']; i += 1
        if not part:   # one read that does not fit at all: skip it
            say(event='skip', key=jobs[i]['key'], pages=jobs[i]['pages']); i += 1; continue
        reqs = [dict(messages=[dict(role='system', content=j['s']['system']), dict(role='user', content=j['s']['user'])], max_tokens=1, seed=0,
                     temperature=0.0, top_k=1, top_p=1.0, think=dict(effort=j['s']['effort'], prefix_token_ids=j['prefix'])) for j in part]
        try:
            res = e.batch(reqs, timeout=1800)
        except Exception as ex:   # noqa
            say(event='batch_failed', error=repr(ex)[:300], keys=[j['key'] for j in part]); time.sleep(10); continue
        for j, r in zip(part, res):
            done[j['key']] = (r['R'], r['next'] if r['next'] is not None else -1)
        flush()
        rate = (len(done) - n0) / max(1, time.time() - t_start)
        say(event='reads', done=len(done), of=n0 + len(jobs), secs=round(res[0]['secs'], 1), eta_min=round((n0 + len(jobs) - len(done)) / max(rate, 1e-6) / 60, 1))
    report(a)


def report(a):
    import numpy as np
    out = Path(a.out); z = np.load(out / 'reads.npz', allow_pickle=False)
    runs = sorted(set(z['run'].tolist())); run_med = [r for r in runs if 'medium' in r][0]; run_fl = [r for r in runs if r != run_med][0]
    R = z['R'].astype(np.float32); pos = z['pos']; run = z['run']
    lines = [f'reads {len(R)}  medium={run_med} flare={run_fl}', '']
    deltas = {}
    for which, mask_pos in (('answer', pos != 'close'), ('close', pos == 'close')):
        m_med = (run == run_med) & mask_pos; m_fl = (run == run_fl) & mask_pos
        mean_med = R[m_med].mean(0); mean_fl = R[m_fl].mean(0)
        delta = mean_med - mean_fl                                   # (64, 5120): add this to a flare-trigger state to move it toward medium
        rms_med = np.sqrt((R[m_med] ** 2).mean(-1)).mean(0); rms_fl = np.sqrt((R[m_fl] ** 2).mean(-1)).mean(0)
        # how separable: projection of each read on the unit delta, per layer, as d' between the two conditions
        u = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-6)
        proj = np.einsum('nld,ld->nl', R, u)
        dprime = (proj[m_med].mean(0) - proj[m_fl].mean(0)) / np.sqrt(0.5 * (proj[m_med].var(0) + proj[m_fl].var(0)) + 1e-6)
        deltas[which] = dict(delta=delta, rms_med=rms_med, rms_fl=rms_fl, dprime=dprime, n_med=int(m_med.sum()), n_fl=int(m_fl.sum()))
        lines.append(f'== {which} positions: medium reads {int(m_med.sum())}, flare reads {int(m_fl.sum())}')
        lines.append('layer  |delta|  |delta|/rms_med  rms_med  rms_fl   dprime')
        for L in range(64):
            lines.append(f'{L:5d}  {np.linalg.norm(delta[L]):7.2f}  {np.linalg.norm(delta[L]) / rms_med[L]:7.3f}  {rms_med[L]:7.2f}  {rms_fl[L]:7.2f}  {dprime[L]:7.2f}')
        lines.append('')
    # cosine of the answer delta with the think direction where both exist
    try:
        T = np.load('/mnt/nvme/persona_weights/concept_cc/think.npz')
        cos = [(int(L), float(deltas['answer']['delta'][L] @ T['d'][j] / (np.linalg.norm(deltas['answer']['delta'][L]) * np.linalg.norm(T['d'][j]) + 1e-9))) for j, L in enumerate(T['layers'].tolist())]
        lines.append('cosine(answer delta, think direction) per layer: ' + ' '.join(f'L{L}:{c:+.2f}' for L, c in cos))
    except Exception as ex:   # noqa
        lines.append(f'think.npz not read: {ex!r}')
    d = deltas['answer']
    np.savez(out / 'delta.npz', layers=np.arange(64), d=d['delta'], n_w=np.linalg.norm(d['delta'], axis=-1), rms_med=d['rms_med'], rms_fl=d['rms_fl'], dprime=d['dprime'],
             n_med=d['n_med'], n_fl=d['n_fl'], source=f'{run_med} minus {run_fl}, answer positions', close_d=deltas['close']['delta'], close_dprime=deltas['close']['dprime'])
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    say(event='delta', path=str(out / 'delta.npz'), top_layers=[int(x) for x in np.argsort(-d['dprime'])[:8]])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prompts'); p.add_argument('--runs', nargs=2, required=True); p.add_argument('--out', required=True); p.add_argument('--ticks', default='2-9')
    p.add_argument('--per-tick', type=int, default=3); p.add_argument('--fourfold', default='/mnt/storage16tb/fourfold/fourfold')
    c = sub.add_parser('capture'); c.add_argument('--engine', required=True); c.add_argument('--out', required=True); c.add_argument('--fracs', default='0.1,0.3,0.5,0.7,0.9')
    c.add_argument('--tokenizer', default='/mnt/nvme/models/Qwen3.8-27B-NVFP4')
    r = sub.add_parser('report'); r.add_argument('--out', required=True)
    a = ap.parse_args()
    {'prompts': prompts, 'capture': capture, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
