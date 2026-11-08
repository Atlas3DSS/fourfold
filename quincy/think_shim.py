#!/usr/bin/env python3
"""OpenAI-style /v1/chat/completions in front of neuropedia's batch thinking server (fabe-integration, 2026-11-08).

The engine (fable_think_server.py / fourfold_think_server.py) takes one synchronous POST per batch of up to four
requests on /v1/batch/completions, with thinking opted in per request by `think={"effort": ...}`. fourfold's sim speaks
plain chat completions with `reasoning_effort` in the body, so this shim:
  - gathers concurrent chat requests for up to --gather seconds (max 4), runs one batch at a time;
  - maps reasoning_effort -> think.effort (default --effort); chat_template_kwargs.enable_thinking=false -> no thinking;
  - preflights the batch and trims max_tokens to what the slots' pages allow;
  - returns message.content = the answer after </think>, message.reasoning_content = the reasoning, finish_reason and
    usage, plus completion_tokens_details.reasoning_tokens is not known (token ids live in the engine's npz artifact);
  - deletes the engine's per-request activation and provenance artifacts unless --keep-artifacts (they are ~7 MB each).

    python3 think_shim.py --engine http://127.0.0.1:8930 --port 9416 [--effort xhigh] [--gather 1.0] [--log shim.jsonl]
"""
import argparse, json, math, os, sys, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

END_THINK, IM_END, EOT, IM_START, THINK = 248069, 248046, 248044, 248045, 248068


def pulse_program(kind, alpha, direction, astra=''):
    """The stop pulse: a push along the think direction (concept_cc think.npz, residual sites L32-48, alpha in n_w units) or along
    Astra-Stop's </think> direction at layer 63 (alpha in residual units). Phase decode: only generated positions are pushed."""
    import numpy as np
    sites = []
    if kind == 'd':
        Z = np.load(direction)
        for j, L in enumerate(Z['layers'].tolist()):
            d = Z['d'][j] / np.linalg.norm(Z['d'][j]); sites.append(dict(site=f'layers.{L}.residual', vector=(d * alpha * float(Z['n_w'][j])).astype(np.float32).tolist()))
    elif kind == 'astra':
        u = np.load(astra, allow_pickle=False).astype(np.float64); u /= np.linalg.norm(u)
        sites.append(dict(site='layers.63.residual', vector=(u * alpha).astype(np.float32).tolist()))
    else:
        raise ValueError(kind)
    return dict(phase='decode', sites=sites)

EFFORTS = ('xhigh', 'medium', 'low')
PASS = ('seed', 'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'steering')


def http(url, payload=None, timeout=7200):
    data = None if payload is None else json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} from {url}: {e.read().decode("utf-8", "replace")[:800]}') from None


class Shim:
    def __init__(self, a):
        self.a = a
        d = http(a.engine + '/v1/models', timeout=60)['data'][0]
        self.model_id = d['id']; self.batch = d['flare']['batch']
        self.page_tokens = self.batch['page_tokens']; self.pages = self.batch['physical_pages']; self.max_batch = self.batch.get('max_batch_size', 4)
        self.context = self.batch.get('logical_context') or 32768
        self.cap = (d['flare'].get('batch', {}).get('thinking') or {}).get('fourfold_max_tokens') or 4096
        self.lock = threading.Condition(); self.pending = []; self.busy = False
        self.log = open(a.log, 'a') if a.log else None
        self.program = json.load(open(a.program)) if a.program else None   # {"steering": {...}, "split": "think"|"window", "window_start": N}
        if self.program:
            print(json.dumps({'event': 'program', 'sites': len(self.program.get('steering', {}).get('sites', [])), 'split': self.program.get('split'), 'window_start': self.program.get('window_start')}), flush=True)
        self.pulse = pulse_program(a.pulse_kind, a.pulse_alpha, a.direction, a.astra) if a.pulse_at else None
        if self.pulse:
            print(json.dumps({'event': 'pulse', 'at': a.pulse_at, 'kind': a.pulse_kind, 'alpha': a.pulse_alpha, 'tokens': a.pulse_tokens, 'answer_tokens': a.answer_tokens}), flush=True)
        self.tok = None
        if a.tokenizer:   # exact prompt token counts, so budgets are trimmed to the page pool before the engine refuses them
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(a.tokenizer)
        print(json.dumps({'event': 'engine', 'model_id': self.model_id, 'batch': {k: self.batch.get(k) for k in ('page_tokens', 'physical_pages', 'max_batch_size', 'qualified')}, 'cap': self.cap}), flush=True)

    def submit(self, body):
        """Blocks until this request's reply is ready; batches with whatever else arrives in the gather window."""
        slot = {'body': body, 'done': threading.Event(), 'reply': None, 'at': time.time()}
        with self.lock:
            self.pending.append(slot)
            self.lock.notify_all()
        slot['done'].wait()
        return slot['reply']

    def loop(self):
        while True:
            with self.lock:
                while not self.pending:
                    self.lock.wait()
                deadline = time.time() + self.a.gather
                while len(self.pending) < self.max_batch and time.time() < deadline:
                    self.lock.wait(timeout=max(0.01, deadline - time.time()))
                part, self.pending = self.pending[:self.max_batch], self.pending[self.max_batch:]
            try:
                self.run(part)
            except Exception as e:  # noqa
                for s in part:
                    s['reply'] = (502, {'error': {'message': f'engine batch failed: {e!r}'[:1500]}}); s['done'].set()

    def prompt_tokens(self, req):
        """Rendered prompt length: the thinking header is the model's own template with enable_thinking=True."""
        if self.tok is None:
            return sum(len(m['content']) for m in req['messages']) // 3 + 64
        kw = {'enable_thinking': False} if 'chat_template_kwargs' in req else {'enable_thinking': True, 'reasoning_effort': req['think']['effort']}
        return len(self.tok.apply_chat_template(req['messages'], tokenize=True, add_generation_prompt=True, return_dict=False, **kw))

    def fit_budgets(self, reqs):
        """Trim every max_tokens so prompt + budget fits the request's share of the page pool (and the context)."""
        share = (self.pages // len(reqs)) * self.page_tokens
        for r in reqs:
            n = self.prompt_tokens(r)
            allowed = min(share, self.context) - n - 1
            if allowed < 16:
                raise ValueError(f'prompt of {n} tokens leaves no room in a {share}-token slot share')
            r['_prompt_tokens'] = n
            if r['max_tokens'] > allowed:
                r['_trimmed_from'] = r['max_tokens']; r['max_tokens'] = allowed
            if '_window_start' in r:   # the gated push: on from prompt + start, to the end of the reply
                r['think']['window'] = [n + r.pop('_window_start'), n + r['max_tokens']]

    def translate(self, body):
        msgs = [{'role': m['role'], 'content': m['content'] if isinstance(m['content'], str) else ''.join(p.get('text', '') for p in m['content'])} for m in body['messages']]
        req = {'model': self.model_id, 'messages': msgs, 'stream': False, 'metadata': {'job_id': 'fourfold-' + uuid.uuid4().hex[:12]}}
        mt = body.get('max_tokens') or body.get('max_completion_tokens') or 2048
        req['max_tokens'] = int(min(int(mt), self.cap))
        if self.a.pulse_at and body.get('reasoning_effort', self.a.effort) != 'off':
            req['max_tokens'] = min(req['max_tokens'], int(self.a.pulse_at))
        for k in PASS:
            if body.get(k) is not None:
                req[k] = body[k]
        req.setdefault('seed', 0)
        if 'temperature' not in req:   # the model card's thinking preset, as the engine's own tests use
            req.update(temperature=1.0, top_k=20, top_p=0.95)
        think_off = (body.get('chat_template_kwargs') or {}).get('enable_thinking') is False
        if think_off:
            req['chat_template_kwargs'] = {'enable_thinking': False}
        else:
            effort = body.get('reasoning_effort') or self.a.effort
            if effort not in EFFORTS:
                raise ValueError(f'reasoning_effort must be one of {EFFORTS}')
            req['think'] = {'effort': effort}
            if self.program and 'steering' not in req and not body.get('no_program'):
                req['steering'] = self.program['steering']
                if self.program.get('split') == 'think':
                    req['think']['split'] = 'think'
                elif self.program.get('split') == 'window':
                    req['think']['split'] = 'window'; req['_window_start'] = int(self.program.get('window_start', 2048))
        return req

    def ids_of(self, item):
        """The engine's generated token ids for one result (from its activation artifact)."""
        import numpy as np
        p = (item['response'].get('activation_artifact') or {}).get('path')
        with np.load(p, allow_pickle=False) as z:
            return z['generated_token_ids'].tolist()

    def pulse_stage(self, live, results):
        """Requests that ran out of thinking budget without closing: a short push along the stop direction (prefix = what they
        thought so far), then an unsteered continuation from the first </think>. Returns {id(req): (text, extra)}."""
        todo = []
        for (s, r), item in zip(live, results):
            if 'think' not in r or item['response']['choices'][0].get('finish_reason') != 'length':
                continue
            ids = [t for t in self.ids_of(item) if t not in (IM_END, EOT)]
            if END_THINK in ids:
                continue
            todo.append(((s, r), item, ids))
        if not todo:
            return {}
        out = {}
        for b in range(0, len(todo), self.max_batch):
            part = todo[b:b + self.max_batch]
            reqs = [dict(model=self.model_id, messages=r['messages'], max_tokens=self.a.pulse_tokens, seed=(r.get('seed') or 0) + 1, stream=False,
                         metadata={'job_id': 'pulse-' + uuid.uuid4().hex[:10]}, think={'effort': r['think']['effort'], 'prefix_token_ids': ids}, steering=self.pulse,
                         **{k: r[k] for k in ('temperature', 'top_k', 'top_p') if k in r}) for (s, r), item, ids in part]
            t0 = time.time(); raw2 = http(self.a.engine + '/v1/batch/completions', {'batch_id': 'ffp-' + uuid.uuid4().hex[:12], 'requests': reqs}, timeout=self.a.timeout)
            cont = []
            for ((s, r), item, ids), item2 in zip(part, raw2['results']):
                pids = self.ids_of(item2); keep = []
                for x in pids:
                    if x == END_THINK:
                        keep.append(x); break
                    if x >= 248000:
                        break
                    keep.append(x)
                if END_THINK not in keep:
                    keep.append(END_THINK)   # the pulse did not close by itself: close it (the forced-cap fallback)
                cont.append(((s, r), ids + keep))
                for key in ('activation_artifact', 'provenance_artifact'):
                    pth = (item2['response'].get(key) or {}).get('path')
                    if pth and self.a.artifact_root and pth.startswith(self.a.artifact_root):
                        try: os.remove(pth)
                        except OSError: pass
            reqs3 = [dict(model=self.model_id, messages=r['messages'], max_tokens=self.a.answer_tokens, seed=(r.get('seed') or 0) + 2, stream=False,
                          metadata={'job_id': 'answer-' + uuid.uuid4().hex[:10]}, think={'effort': r['think']['effort'], 'prefix_token_ids': prefix},
                          **{k: r[k] for k in ('temperature', 'top_k', 'top_p') if k in r}) for (s, r), prefix in cont]
            raw3 = http(self.a.engine + '/v1/batch/completions', {'batch_id': 'ffa-' + uuid.uuid4().hex[:12], 'requests': reqs3}, timeout=self.a.timeout)
            secs = time.time() - t0
            for ((s, r), prefix), item3 in zip(cont, raw3['results']):
                aids = [t for t in self.ids_of(item3) if t not in (IM_END, EOT)]
                think_ids = prefix[:prefix.index(END_THINK)]
                reasoning = self.tok.decode(think_ids, skip_special_tokens=True) if self.tok else ''
                answer = item3['response']['choices'][0]['message'].get('content') or ''
                text = reasoning + '</think>' + answer
                out[id(r)] = (text, {'finish': item3['response']['choices'][0].get('finish_reason'),
                                     'usage': {'pulsed': True, 'pulse_at': len(think_ids), 'completion_tokens': len(prefix) + len(aids), 'reasoning_tokens': len(think_ids), 'answer_tokens': len(aids), 'pulse_seconds': round(secs, 1)}})
                for key in ('activation_artifact', 'provenance_artifact'):
                    pth = (item3['response'].get(key) or {}).get('path')
                    if pth and self.a.artifact_root and pth.startswith(self.a.artifact_root):
                        try: os.remove(pth)
                        except OSError: pass
        return out

    def run(self, part):
        reqs = []
        for s in part:
            try:
                reqs.append(self.translate(s['body']))
            except Exception as e:  # noqa
                s['reply'] = (400, {'error': {'message': str(e)}}); s['done'].set(); reqs.append(None)
        live = [(s, r) for s, r in zip(part, reqs) if r is not None]
        if not live:
            return
        self.fit_budgets([r for _, r in live])
        meta = {id(r): {k: r.pop(k) for k in ('_prompt_tokens', '_trimmed_from') if k in r} for _, r in live}
        batch = {'batch_id': 'ff-' + uuid.uuid4().hex[:12], 'requests': [r for _, r in live]}
        pre = http(self.a.engine + '/v1/batch/preflight', batch, timeout=300)
        if pre.get('fits') is not True:
            raise RuntimeError('preflight refused: ' + str(pre.get('reason')))
        t0 = time.time()
        raw = http(self.a.engine + '/v1/batch/completions', batch, timeout=self.a.timeout)
        secs = time.time() - t0
        results = raw.get('results') or []
        if len(results) != len(live):
            raise RuntimeError(f'batch coverage differs: {len(results)} results for {len(live)} requests')
        pulsed = {}
        if self.pulse:
            pulsed = self.pulse_stage(live, results)
        for (s, r), item in zip(live, results):
            resp = item['response']; choice = resp['choices'][0]; text = choice['message'].get('content') or ''
            if id(r) in pulsed:
                text, extra = pulsed[id(r)]; choice = dict(choice, finish_reason=extra['finish']); resp = dict(resp, usage={**(resp.get('usage') or {}), **extra['usage']})
            if '</think>' in text:
                reasoning, answer = text.split('</think>', 1); reasoning = reasoning.strip(); answer = answer.lstrip('\n')
            elif 'think' in r:
                reasoning, answer = text, ''
            else:
                reasoning, answer = None, text
            usage = resp.get('usage') or {}
            out = {'id': resp.get('id') or 'chatcmpl-' + uuid.uuid4().hex[:12], 'object': 'chat.completion', 'created': int(time.time()), 'model': s['body'].get('model') or self.model_id,
                   'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer, 'reasoning_content': reasoning}, 'finish_reason': choice.get('finish_reason')}],
                   'usage': {**usage, 'batch_seconds': round(secs, 1), 'batch_size': len(live), 'think_effort': (r.get('think') or {}).get('effort'), 'max_tokens_sent': r['max_tokens'], **meta.get(id(r), {})}}
            s['reply'] = (200, out); s['done'].set()
            if self.log:
                self.log.write(json.dumps({'at': time.time(), 'batch': batch['batch_id'], 'n': len(live), 'secs': round(secs, 1), 'finish': choice.get('finish_reason'), 'usage': usage,
                                           'reasoning_chars': len(reasoning or ''), 'answer_chars': len(answer), 'effort': (r.get('think') or {}).get('effort'), 'max_tokens': r['max_tokens']}) + '\n'); self.log.flush()
            if not self.a.keep_artifacts:
                for key in ('activation_artifact', 'provenance_artifact'):
                    p = (resp.get(key) or {}).get('path')
                    if p and self.a.artifact_root and p.startswith(self.a.artifact_root):
                        try: os.remove(p)
                        except OSError: pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--engine', required=True); ap.add_argument('--port', type=int, required=True); ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--effort', default='xhigh', choices=EFFORTS); ap.add_argument('--gather', type=float, default=1.0); ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--log', default=''); ap.add_argument('--tokenizer', default='', help='tokenizer dir for exact prompt counts (the model checkpoint)'); ap.add_argument('--program', default='', help='json: steering program (+ split/window_start) attached to every thinking request');
    ap.add_argument('--pulse-at', type=int, default=0, help='thinking allowance in tokens; a reply still thinking at this point gets the stop pulse, then an unsteered answer'); ap.add_argument('--pulse-kind', default='d', choices=('d', 'astra')); ap.add_argument('--direction', default='', help='think-direction .npz (layers, d, n_w)'); ap.add_argument('--astra', default='', help="Astra-Stop u.npy"); ap.add_argument('--pulse-alpha', type=float, default=4.0); ap.add_argument('--pulse-tokens', type=int, default=3); ap.add_argument('--answer-tokens', type=int, default=1500); ap.add_argument('--keep-artifacts', action='store_true'); ap.add_argument('--artifact-root', default='', help='delete the engine artifacts under this directory after each reply (empty: keep them)')
    a = ap.parse_args()
    shim = Shim(a)
    threading.Thread(target=shim.loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, code, obj):
            data = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path.startswith('/v1/models'):
                self.send(200, {'object': 'list', 'data': [{'id': 'qwen38-27b', 'object': 'model', 'owned_by': 'fourfold-think-shim', 'engine_model_id': shim.model_id,
                                                             'flare': {'batch': shim.batch, 'max_concurrency': shim.max_batch, 'shim': {'effort': a.effort, 'cap': shim.cap}}}]})
            else:
                self.send(404, {'error': {'message': 'not found'}})
        def do_POST(self):
            if not self.path.startswith('/v1/chat/completions'):
                return self.send(404, {'error': {'message': 'only /v1/chat/completions'}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                if body.get('stream'):
                    return self.send(400, {'error': {'message': 'streaming is not supported'}})
                code, reply = shim.submit(body)
            except Exception as e:  # noqa
                code, reply = 500, {'error': {'message': repr(e)[:800]}}
            self.send(code, reply)

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(json.dumps({'event': 'listening', 'port': a.port, 'engine': a.engine}), flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
