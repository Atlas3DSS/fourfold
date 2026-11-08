#!/usr/bin/env python3
"""OpenAI chat endpoint in front of a Flare steering server that adds the stop trigger (fabe-integration, 2026-11-08).

Each chat request becomes: (1) the same chat request to Flare with max_tokens capped at the thinking allowance; (2) if it
ended by length still inside the think block, a raw continuation of the rendered prompt + reasoning with a steering
program along the think direction for --pulse-tokens tokens (phase decode: generated positions only); (3) a raw, unsteered
continuation from the first </think> for the answer. The reply carries the answer in content and the reasoning in
reasoning_content, so the sim logs both; usage says whether it was pulsed. Flare handles concurrency itself.

    python3 flare_pulse_shim.py --flare http://127.0.0.1:9426/v1 --port 9436 --pulse-at 2048 [--pulse-alpha 4] [--pulse-tokens 3] [--answer-tokens 1500]
"""
import argparse, json, re, sys, threading, time, urllib.request, uuid
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


def render(messages, effort, assistant_prefix):
    """Qwen3.8 ChatML for the raw continuation: the effort sentence goes first in the system turn, as the template does."""
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system'); eff = EFFORT_TEXT.get(effort, '')
    sysx = (eff + '\n\n' + system) if eff and system else (eff or system)
    out = f"<|im_start|>system\n{sysx}<|im_end|>\n" if sysx else ''
    for m in messages:
        if m['role'] in ('user', 'assistant'):
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + '<|im_start|>assistant\n' + assistant_prefix


class Shim:
    def __init__(self, a):
        self.a = a; self.prog = program(a.pulse_kind, a.pulse_alpha, a.direction, a.astra); self.log = open(a.log, 'a') if a.log else None

    def handle(self, body):
        a = self.a; messages = body['messages']; effort = body.get('reasoning_effort') or a.effort
        think_off = (body.get('chat_template_kwargs') or {}).get('enable_thinking') is False
        req = {k: v for k, v in body.items() if k not in ('stream',)}; req['model'] = body.get('model') or 'qwen38-27b'
        if not think_off:
            req['max_tokens'] = min(int(body.get('max_tokens') or body.get('max_completion_tokens') or 8192), a.pulse_at); req['reasoning_effort'] = effort
        req.pop('max_completion_tokens', None)
        t0 = time.time(); r1 = post(a.flare + '/chat/completions', req, a.timeout)
        ch = r1['choices'][0]; msg = ch['message']; reasoning = msg.get('reasoning_content') or ''; content = msg.get('content') or ''
        usage = dict(r1.get('usage') or {}); usage['pulsed'] = False
        if not think_off and ch.get('finish_reason') == 'length' and not content.strip():
            seed = int(body.get('seed') or 0)
            prefix = render(messages, effort, '<think>\n' + reasoning)
            r2 = post(a.flare + '/completions', dict(model=req['model'], prompt=prefix, max_tokens=a.pulse_tokens, seed=seed + 1, steering=self.prog), a.timeout)
            pulse = r2['choices'][0].get('text') or ''
            closed = '</think>' in pulse
            cut = (pulse.split('</think>', 1)[0] + '</think>') if closed else (pulse + '</think>')
            r3 = post(a.flare + '/completions', dict(model=req['model'], prompt=prefix + cut + '\n\n', max_tokens=a.answer_tokens, seed=seed + 2), a.timeout)
            content = r3['choices'][0].get('text') or ''; reasoning = reasoning + cut.split('</think>')[0]
            u3 = r3.get('usage') or {}
            usage.update(pulsed=True, pulse_closed=closed, pulse_at=usage.get('completion_tokens'), answer_tokens=u3.get('completion_tokens'),
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
    ap.add_argument('--effort', default='xhigh'); ap.add_argument('--pulse-at', type=int, default=2048); ap.add_argument('--pulse-kind', default='d'); ap.add_argument('--direction', required=True, help='think-direction .npz (layers, d, n_w)'); ap.add_argument('--astra', default='', help="Astra-Stop u.npy (for --pulse-kind astra)"); ap.add_argument('--pulse-alpha', type=float, default=4.0)
    ap.add_argument('--pulse-tokens', type=int, default=3); ap.add_argument('--answer-tokens', type=int, default=1500); ap.add_argument('--timeout', type=int, default=3600); ap.add_argument('--log', default='')
    a = ap.parse_args(); shim = Shim(a)
    print(json.dumps({'event': 'listening', 'port': a.port, 'flare': a.flare, 'pulse_at': a.pulse_at, 'kind': a.pulse_kind, 'alpha': a.pulse_alpha}), flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, code, obj):
            data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path.startswith('/v1/models'):
                self.send(200, {'object': 'list', 'data': [{'id': 'qwen38-27b', 'object': 'model', 'owned_by': 'fourfold-flare-pulse-shim', 'flare': {'max_concurrency': 4, 'shim': {'pulse_at': a.pulse_at, 'kind': a.pulse_kind, 'alpha': a.pulse_alpha}}}]})
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
