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

    def translate(self, body):
        msgs = [{'role': m['role'], 'content': m['content'] if isinstance(m['content'], str) else ''.join(p.get('text', '') for p in m['content'])} for m in body['messages']]
        req = {'model': self.model_id, 'messages': msgs, 'stream': False, 'metadata': {'job_id': 'fourfold-' + uuid.uuid4().hex[:12]}}
        mt = body.get('max_tokens') or body.get('max_completion_tokens') or 2048
        req['max_tokens'] = int(min(int(mt), self.cap))
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
        return req

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
        for (s, r), item in zip(live, results):
            resp = item['response']; choice = resp['choices'][0]; text = choice['message'].get('content') or ''
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
    ap.add_argument('--log', default=''); ap.add_argument('--tokenizer', default='', help='tokenizer dir for exact prompt counts (the model checkpoint)'); ap.add_argument('--keep-artifacts', action='store_true'); ap.add_argument('--artifact-root', default='', help='delete the engine artifacts under this directory after each reply (empty: keep them)')
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
