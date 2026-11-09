#!/usr/bin/env python3
"""The stop-trigger shim (flare_pulse_shim.py) plus an answer lace: a steering vector added during the answer (fabe-integration,
2026-11-08, owner's question "does the pulse make them not want to act?").

Same three steps as the pulse shim: (1) chat request capped at the thinking allowance; (2) if still thinking, a raw continuation
with the 3-token pulse along the think direction; (3) a raw continuation for the answer. Step 3 now carries a steering program
too: --answer-direction is a delta file in the think.npz layout (layers, d, n_w), here delta_capture.py's medium-minus-flare
residual delta; at each chosen layer unit(d) * alpha * n_w is added at layers.L.residual, phase decode, for every answer token.
With alpha 1 and n_w = |delta| that is exactly the measured mean shift. Replies that closed on their own (not pulsed) get no lace.

    python3 flare_pulse_shim2.py --flare http://127.0.0.1:9426/v1 --port 9440 --pulse-at 2048 --direction think.npz \
        --answer-direction delta.npz --answer-layers 24-56:4 --answer-alpha 1.0
"""
import argparse, json, math, re, sys, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

EFFORT_TEXT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.', 'medium': ''}


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} from Flare: {e.read().decode("utf-8", "replace")[:500]}') from None


def parse_layers(spec):
    """'24-56:4' -> 24,28,...,56; '32,40,48' -> those; '' -> every layer in the file."""
    if not spec:
        return None
    out = []
    for part in spec.split(','):
        if '-' in part:
            rng, _, step = part.partition(':'); lo, hi = (int(x) for x in rng.split('-')); out.extend(range(lo, hi + 1, int(step or 1)))
        else:
            out.append(int(part))
    return out


def program(direction, alpha, layers=None):
    """think.npz layout: layers, d (layers, hidden), n_w (layers). vector = unit(d) * alpha * n_w at layers.L.residual."""
    Z = np.load(direction); sites = []
    file_layers = Z['layers'].tolist()
    for j, L in enumerate(file_layers):
        if layers is not None and L not in layers:
            continue
        d = Z['d'][j].astype(np.float64); n = np.linalg.norm(d)
        if n == 0:
            continue
        sites.append(dict(site=f'layers.{L}.residual', vector=(d / n * alpha * float(Z['n_w'][j])).astype(np.float32).tolist()))
    return dict(phase='decode', sites=sites), [int(s['site'].split('.')[1]) for s in sites], [round(float(np.linalg.norm(s['vector'])), 2) for s in sites]


def bounce(x, peak0, rate0, floor=0.05, rise_to=0.95):
    """owner's schedule (2026-11-08): start at peak0, fall off exponentially at rate0; at the floor (5% of peak0) bounce off zero
    and rise toward half the previous peak at half the previous rate, then fall again at that rate; rinse and repeat.
    x is the position on the chosen axis (answer tokens, or ticks) counted from 0."""
    peak, rate, x0, phase, a = peak0, rate0, 0, 'fall', peak0
    for i in range(1, int(x) + 1):
        if phase == 'fall':
            a = peak * math.exp(-rate * (i - x0))
            if a <= floor * peak0:
                peak, rate, x0, phase, a = peak / 2, rate / 2, i, 'rise', 0.0
        else:
            a = peak * (1 - math.exp(-rate * (i - x0)))
            if a >= rise_to * peak:
                x0, phase = i, 'fall'
    return a


def parse_schedule(spec):
    """'bounce:0.825:0.03' -> (kind, peak, rate); '' -> None."""
    if not spec:
        return None
    kind, peak, rate = spec.split(':'); return kind, float(peak), float(rate)


def render(messages, effort, assistant_prefix):
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system'); eff = EFFORT_TEXT.get(effort, '')
    sysx = (eff + '\n\n' + system) if eff and system else (eff or system)
    out = f"<|im_start|>system\n{sysx}<|im_end|>\n" if sysx else ''
    for m in messages:
        if m['role'] in ('user', 'assistant'):
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + '<|im_start|>assistant\n' + assistant_prefix


class Shim:
    def __init__(self, a):
        self.a = a
        self.pulse, pl, pn = program(a.direction, a.pulse_alpha)
        self.answer, al, an = (program(a.answer_direction, a.answer_alpha, parse_layers(a.answer_layers)) if a.answer_direction else (None, [], []))
        self.schedule = parse_schedule(a.answer_schedule)
        self.unit = [(s['site'], np.array(s['vector'], dtype=np.float32) / max(a.answer_alpha, 1e-9)) for s in (self.answer or {}).get('sites', [])]  # alpha 1 vectors
        self.info = dict(pulse_layers=pl, pulse_norms=pn, answer_layers=al, answer_norms=an, answer_alpha=a.answer_alpha, answer_lace_tokens=a.answer_lace_tokens,
                         answer_schedule=a.answer_schedule, answer_axis=a.answer_axis, answer_segment=a.answer_segment)
        self.log = open(a.log, 'a') if a.log else None

    def scaled(self, alpha):
        return dict(phase='decode', sites=[dict(site=s, vector=(v * alpha).tolist()) for s, v in self.unit])

    def alpha_at(self, x):
        kind, peak, rate = self.schedule
        return bounce(x, peak, rate) if kind == 'bounce' else peak

    def answer_scheduled(self, r3body, seed, tick):
        """the answer with the dose following the schedule: per tick (one call) or per answer token (segments)."""
        a = self.a; alphas = []
        if a.answer_axis == 'tick':
            alpha = self.alpha_at(max(0, tick - 1)); alphas.append(round(alpha, 4))
            body = dict(r3body); body['steering'] = self.scaled(alpha) if alpha > 0.01 else None
            if body['steering'] is None: body.pop('steering')
            r3 = post(a.flare + '/completions', body, a.timeout); return r3, alphas
        text, used, finish, k = '', 0, 'length', 0
        while used < a.answer_tokens:
            n = min(a.answer_segment, a.answer_tokens - used)
            alpha = self.alpha_at(used + n // 2); alphas.append(round(alpha, 4))
            body = dict(r3body, prompt=r3body['prompt'] + text, max_tokens=n, seed=seed + 2 + k)
            if alpha > 0.01: body['steering'] = self.scaled(alpha)
            r = post(a.flare + '/completions', body, a.timeout); k += 1
            piece = r['choices'][0].get('text') or ''; text += piece; used += int((r.get('usage') or {}).get('completion_tokens') or n)
            finish = r['choices'][0].get('finish_reason')
            if finish != 'length' or not piece:
                break
        return {'choices': [{'text': text, 'finish_reason': finish}], 'usage': {'completion_tokens': used}}, alphas

    def handle(self, body):
        a = self.a; messages = body['messages']; effort = body.get('reasoning_effort') or a.effort
        think_off = (body.get('chat_template_kwargs') or {}).get('enable_thinking') is False
        req = {k: v for k, v in body.items() if k not in ('stream',)}; req['model'] = body.get('model') or 'qwen38-27b'
        if not think_off:
            req['max_tokens'] = min(int(body.get('max_tokens') or body.get('max_completion_tokens') or 8192), a.pulse_at); req['reasoning_effort'] = effort
        req.pop('max_completion_tokens', None)
        t0 = time.time(); r1 = post(a.flare + '/chat/completions', req, a.timeout)
        ch = r1['choices'][0]; msg = ch['message']; reasoning = msg.get('reasoning_content') or ''; content = msg.get('content') or ''
        usage = dict(r1.get('usage') or {}); usage['pulsed'] = False; usage['answer_laced'] = False
        if not think_off and ch.get('finish_reason') == 'length' and not content.strip():
            seed = int(body.get('seed') or 0)
            prefix = render(messages, effort, '<think>\n' + reasoning)
            r2 = post(a.flare + '/completions', dict(model=req['model'], prompt=prefix, max_tokens=a.pulse_tokens, seed=seed + 1, steering=self.pulse), a.timeout)
            pulse = r2['choices'][0].get('text') or ''
            closed = '</think>' in pulse
            cut = (pulse.split('</think>', 1)[0] + '</think>') if closed else (pulse + '</think>')
            r3body = dict(model=req['model'], prompt=prefix + cut + '\n\n', max_tokens=a.answer_tokens, seed=seed + 2)
            alphas = None
            if self.answer and self.schedule:                    # the owner's bouncing dose, over answer tokens or over ticks
                m = re.search(r'^tick (\d+)', '\n'.join(m['content'] for m in messages if m['role'] == 'user'), re.M)
                r3, alphas = self.answer_scheduled(r3body, seed, int(m.group(1)) if m else 1)
            elif self.answer and a.answer_lace_tokens > 0:        # a short lace: the delta on the first K answer tokens only, then plain
                r3a = post(a.flare + '/completions', dict(r3body, max_tokens=a.answer_lace_tokens, steering=self.answer), a.timeout)
                head = r3a['choices'][0].get('text') or ''
                r3 = post(a.flare + '/completions', dict(r3body, prompt=r3body['prompt'] + head, max_tokens=a.answer_tokens - a.answer_lace_tokens, seed=seed + 3), a.timeout)
                r3['choices'][0]['text'] = head + (r3['choices'][0].get('text') or '')
                r3['usage'] = dict(r3.get('usage') or {}, completion_tokens=((r3.get('usage') or {}).get('completion_tokens') or 0) + ((r3a.get('usage') or {}).get('completion_tokens') or 0))
            else:
                if self.answer:
                    r3body['steering'] = self.answer
                r3 = post(a.flare + '/completions', r3body, a.timeout)
            content = r3['choices'][0].get('text') or ''; reasoning = reasoning + cut.split('</think>')[0]
            u3 = r3.get('usage') or {}
            if alphas is not None:
                usage['answer_alphas'] = alphas
            usage.update(pulsed=True, pulse_closed=closed, pulse_at=usage.get('completion_tokens'), answer_tokens=u3.get('completion_tokens'), answer_laced=bool(self.answer),
                         completion_tokens=(usage.get('completion_tokens') or 0) + a.pulse_tokens + (u3.get('completion_tokens') or 0))
            ch = dict(ch, finish_reason=r3['choices'][0].get('finish_reason'))
        usage['shim_seconds'] = round(time.time() - t0, 1)
        out = {'id': r1.get('id') or 'chatcmpl-' + uuid.uuid4().hex[:12], 'object': 'chat.completion', 'created': int(time.time()), 'model': req['model'],
               'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content, 'reasoning_content': reasoning or None}, 'finish_reason': ch.get('finish_reason')}], 'usage': usage}
        if self.log:
            self.log.write(json.dumps({'at': time.time(), 'usage': usage, 'reasoning_chars': len(reasoning), 'answer_chars': len(content), 'effort': effort}) + '\n'); self.log.flush()
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--flare', required=True); ap.add_argument('--port', type=int, required=True); ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--effort', default='xhigh'); ap.add_argument('--pulse-at', type=int, default=2048); ap.add_argument('--direction', required=True, help='think-direction .npz (layers, d, n_w)')
    ap.add_argument('--pulse-alpha', type=float, default=4.0); ap.add_argument('--pulse-tokens', type=int, default=3); ap.add_argument('--answer-tokens', type=int, default=1500)
    ap.add_argument('--answer-direction', default='', help='delta .npz (layers, d, n_w) added during the answer'); ap.add_argument('--answer-alpha', type=float, default=1.0)
    ap.add_argument('--answer-layers', default='', help="e.g. '24-56:4' or '32,40,48'; empty = every layer in the file")
    ap.add_argument('--answer-lace-tokens', type=int, default=0, help='>0: add the delta only on the first K answer tokens, then continue plain')
    ap.add_argument('--answer-schedule', default='', help="dose schedule, e.g. 'bounce:0.825:0.03' (kind:peak:rate); overrides --answer-alpha and --answer-lace-tokens")
    ap.add_argument('--answer-axis', default='tokens', choices=['tokens', 'tick'], help='the schedule runs over answer tokens (segments) or over the tick number in the prompt')
    ap.add_argument('--answer-segment', type=int, default=64, help='tokens per segment when the schedule runs over answer tokens')
    ap.add_argument('--timeout', type=int, default=3600); ap.add_argument('--log', default='')
    a = ap.parse_args(); shim = Shim(a)
    print(json.dumps({'event': 'listening', 'port': a.port, 'flare': a.flare, 'pulse_at': a.pulse_at, 'pulse_alpha': a.pulse_alpha, **shim.info}), flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, code, obj):
            data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path.startswith('/v1/models'):
                self.send(200, {'object': 'list', 'data': [{'id': 'qwen38-27b', 'object': 'model', 'owned_by': 'fourfold-flare-pulse-shim2',
                                                            'flare': {'max_concurrency': 4, 'shim': {'pulse_at': a.pulse_at, 'pulse_alpha': a.pulse_alpha, **shim.info}}}]})
            else:
                self.send(404, {'error': {'message': 'not found'}})
        def do_POST(self):
            if not self.path.startswith('/v1/chat/completions'):
                return self.send(404, {'error': {'message': 'only /v1/chat/completions'}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                if body.get('stream'):
                    return self.send(400, {'error': {'message': 'streaming is not supported'}})
                self.send(200, shim.handle(body))
            except Exception as e:  # noqa
                self.send(502, {'error': {'message': repr(e)[:800]}})

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()
