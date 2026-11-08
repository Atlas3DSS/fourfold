#!/usr/bin/env python3
"""The stop trigger on a Flare steering server (fabe-integration, 2026-11-08): the same three steps as the batch-engine pulse,
through Flare's OpenAI endpoints. (1) /v1/chat/completions with the effort, max_tokens N (the thinking allowance); (2) if the
reply ended by length still thinking: /v1/completions on the rendered prompt + reasoning so far with a steering program
(phase decode: generated positions only), max_tokens 3, along the think direction; (3) /v1/completions continuing from the
first </think>, unsteered, for the answer. Runs on Quincy with the Flare venv python (numpy for the direction file).

    $PY think_pulse_flare.py --url http://127.0.0.1:9426/v1 --prompts think_prompts.json --out DIR --pulse-at 2048 --seeds 3
"""
import argparse, json, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

EFFORT_TEXT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.', 'medium': ''}
SUM_EQ = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:500]}') from None


def program(kind, alpha, direction, astra=''):
    """direction: the think-direction file (layers, d, n_w per layer); astra: Astra-Stop's u.npy."""
    sites = []
    if kind == 'd':
        Z = np.load(direction)
        for j, L in enumerate(Z['layers'].tolist()):
            d = Z['d'][j] / np.linalg.norm(Z['d'][j]); sites.append(dict(site=f'layers.{L}.residual', vector=(d * alpha * float(Z['n_w'][j])).astype(np.float32).tolist()))
    elif kind == 'astra':
        u = np.load(astra, allow_pickle=False).astype(np.float64); u /= np.linalg.norm(u)
        sites.append(dict(site='layers.63.residual', vector=(u * alpha).astype(np.float32).tolist()))
    return dict(phase='decode', sites=sites)


def render(system, user, effort, assistant_prefix):
    eff = EFFORT_TEXT.get(effort, ''); sysx = (eff + '\n\n' + system) if eff else system
    return f"<|im_start|>system\n{sysx}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"


def answer_format(text):
    qs = [re.search(rf"<q{i}>\s*([0-9a-fA-F\s_:x]+?)\s*</q{i}>", text, re.S) for i in range(4)]
    hexok = sum(1 for m in qs if m and re.fullmatch(r"[0-9a-f]{32}", re.sub(r"[\s_:]", "", m.group(1)).lower().removeprefix("0x")))
    return dict(has_log=bool(re.search(r"<log>.*?</log>", text, re.S)), q_tags=sum(1 for m in qs if m), q_hex_ok=hexok)


def check_sums(text):
    n = wrong = 0
    for lhs, rhs in SUM_EQ.findall(text):
        try:
            v = eval(re.sub(r"[^\d+]", "", lhs)); n += 1; wrong += v != int(rhs.replace(",", ""))  # noqa: S307
        except Exception:
            pass
    return n, wrong


def one(a, pname, p, seed, prog):
    msgs = [{'role': 'system', 'content': p['system']}, {'role': 'user', 'content': p['user']}]
    t0 = time.time(); row = dict(prompt=pname, seed=seed, effort=a.effort, pulse_at=a.pulse_at)
    r1, s1 = post(a.url + '/chat/completions', dict(model='qwen38-27b', messages=msgs, max_tokens=a.pulse_at, seed=seed, reasoning_effort=a.effort))
    ch = r1['choices'][0]; msg = ch['message']; reasoning = msg.get('reasoning_content') or ''; content = msg.get('content') or ''
    row.update(stage1_finish=ch['finish_reason'], stage1_tokens=(r1.get('usage') or {}).get('completion_tokens'), secs1=round(s1, 1))
    if ch['finish_reason'] == 'length' and not content.strip():
        prefix = render(p['system'], p['user'], a.effort, '<think>\n' + reasoning)
        r2, s2 = post(a.url + '/completions', dict(model='qwen38-27b', prompt=prefix, max_tokens=a.pulse_tokens, seed=seed + 1, steering=prog))
        pulse = r2['choices'][0].get('text') or ''
        closed_in_pulse = '</think>' in pulse
        cut = pulse.split('</think>', 1)[0] + '</think>' if closed_in_pulse else pulse + '</think>'   # the fallback: close it ourselves
        r3, s3 = post(a.url + '/completions', dict(model='qwen38-27b', prompt=prefix + cut + '\n\n', max_tokens=a.answer_tokens, seed=seed + 2))
        answer = r3['choices'][0].get('text') or ''
        row.update(pulsed=True, pulse_text=pulse[:120], closed_in_pulse=closed_in_pulse, secs2=round(s2, 1), secs3=round(s3, 1), answer_tokens=(r3.get('usage') or {}).get('completion_tokens'), finish3=r3['choices'][0].get('finish_reason'))
    else:
        row.update(pulsed=False, closed_in_pulse=None); answer = content
    row.update(seconds=round(time.time() - t0, 1), answer=answer_format(answer), sums=check_sums(answer), content=answer[:2500])
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--url', required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pulse-at', type=int, default=2048); ap.add_argument('--pulse-kind', default='d'); ap.add_argument('--direction', required=True, help='think-direction .npz (layers, d, n_w)'); ap.add_argument('--astra', default='', help="Astra-Stop u.npy (for --pulse-kind astra)"); ap.add_argument('--pulse-alpha', type=float, default=4.0); ap.add_argument('--pulse-tokens', type=int, default=3)
    ap.add_argument('--answer-tokens', type=int, default=1500); ap.add_argument('--effort', default='xhigh'); ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--seed-base', type=int, default=20261108)
    ap.add_argument('--only', default=''); ap.add_argument('--concurrency', type=int, default=3)
    a = ap.parse_args(); prompts = json.load(open(a.prompts)); Path(a.out).mkdir(parents=True, exist_ok=True); prog = program(a.pulse_kind, a.pulse_alpha, a.direction, a.astra)
    import threading
    jobs = [(pn, p, a.seed_base + s) for pn, p in prompts.items() if not a.only or pn in a.only.split(',') for s in range(a.seeds)]
    lock = threading.Lock(); sem = threading.Semaphore(a.concurrency); fh = open(Path(a.out) / 'rows.jsonl', 'a')
    def run(job):
        with sem:
            try:
                row = one(a, job[0], job[1], job[2], prog)
            except Exception as e:  # noqa
                row = dict(prompt=job[0], seed=job[2], error=str(e)[:400])
            with lock:
                fh.write(json.dumps(row) + '\n'); fh.flush()
                print(json.dumps({k: v for k, v in row.items() if k not in ('content',)}), flush=True)
    ts = [threading.Thread(target=run, args=(j,)) for j in jobs]
    for t in ts: t.start()
    for t in ts: t.join()


if __name__ == '__main__':
    main()
