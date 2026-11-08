#!/usr/bin/env python3
"""Thinking-control probe for fourfold agents on Qwen3.8-27B (Flare). fabe-integration, 2026-11-08. Stdlib only.

Sends the same agent prompts under several request-side arms (effort level, presence penalty, a client-side thinking
budget, a steering vector) and records, per reply: tokens out, seconds, whether it still answers in the fourfold
format, how long the reasoning is, whether the reasoning loops (zlib compressibility of 3,000-char windows, the
neuropedia long.py metric: prose ~2.1x, a loop >3.5x), pivot words, and whether every explicit "a + b = c" it writes
is right.

    python3 think_probe.py --url http://127.0.0.1:9406/v1 --prompts think_prompts.json --out DIR \
        --arms arms.json [--seeds 4] [--max-tokens 24576] [--concurrency 4] [--only tick1-agent0,tick64-agent107]
arms.json: [{"name": "xhigh", "effort": "xhigh", "body": {...extra body...}, "budget": null|int, "budget_text": "..."} ...]
"""
import argparse, json, re, sys, threading, time, urllib.request, zlib
from pathlib import Path

THINK_END_TEXT = "</think>"
DEFAULT_BUDGET_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution based on my "
                       "thinking directly now.\n</think>\n\n")
PIVOTS = ("wait", "hmm", "actually", "but what if", "let me re", "alternatively", "hold on")


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        out = {"error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:  # noqa
        out = {"error": repr(e)[:500]}
    return out, time.time() - t0


def compressibility(s):
    b = s.encode()
    return len(b) / max(1, len(zlib.compress(b, 9)))


def loop_stats(text, window=3000, ratio=3.5):
    """(first char index where a window compresses > ratio, or None; max window compressibility; tail compressibility)"""
    if len(text) < 400:
        return None, compressibility(text) if text else 1.0, compressibility(text) if text else 1.0
    best, onset = 0.0, None
    for i in range(0, max(1, len(text) - window + 1), 500):
        c = compressibility(text[i:i + window])
        best = max(best, c)
        if c > ratio and onset is None:
            onset = i
    return onset, round(best, 2), round(compressibility(text[-window:]), 2)


def ngram_repeat(text, n=8):
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 3)


EQ = re.compile(r"(?<![\d.])(\d[\d,]{0,6}(?:\s*[+\-*]\s*\d[\d,]{0,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")
# the task's own arithmetic: sums of two-digit-or-longer numbers (hex digit counting like "5-8=4" is excluded)
SUM_EQ = re.compile(r"(?<![\d.])(\d{2,6}(?:\s*\+\s*\d{2,6}){1,12})\s*=\s*(\d[\d,]{0,8})(?![\d.])")


def check_equations(text, pattern=EQ):
    """every explicit integer a + b (- c) (* d) = e in the text: (n_checked, n_wrong, wrong examples)"""
    n = wrong = 0
    bad = []
    for lhs, rhs in pattern.findall(text):
        try:
            v = eval(re.sub(r"[^\d+\-*]", "", lhs))  # digits and + - * only  # noqa: S307
            r = int(rhs.replace(",", ""))
        except Exception:
            continue
        n += 1
        if v != r:
            wrong += 1
            if len(bad) < 5:
                bad.append(f"{lhs}={rhs} (is {v})")
    return n, wrong, bad


def parse_answer(content):
    """fourfold format: <log>, optional <protocol>, <q0>..<q3> hex"""
    text = content or ""
    if THINK_END_TEXT in text:
        text = text.rsplit(THINK_END_TEXT, 1)[1]
    qs = [re.search(rf"<q{i}>\s*([0-9a-fA-F\s_:x]+?)\s*</q{i}>", text, re.S) for i in range(4)]
    hexok = sum(1 for m in qs if m and re.fullmatch(r"[0-9a-f]{32}", re.sub(r"[\s_:]", "", m.group(1)).lower().removeprefix("0x")))
    return {"has_log": bool(re.search(r"<log>.*?</log>", text, re.S)), "q_tags": sum(1 for m in qs if m), "q_hex_ok": hexok,
            "answer_chars": len(text.strip())}


def render_chatml(system, user, assistant_prefix):
    """Qwen3.8 ChatML with the assistant turn continued (for /v1/completions): effort instruction goes in the system text."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}")


EFFORT_TEXT = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
    "medium": "",
}


def one(a, url, pname, prompt, arm, seed, max_tokens, timeout, out_lock, fh):
    body = {"model": "qwen38-27b", "messages": [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}],
            "max_tokens": max_tokens, "seed": seed}
    if arm.get("effort"):
        body["reasoning_effort"] = arm["effort"]
    body.update(arm.get("body") or {})
    row = {"prompt": pname, "arm": arm["name"], "seed": seed, "at": time.time()}
    budget = arm.get("budget")
    reasoning, content, usage, secs, finish, err, forced = "", "", {}, 0.0, None, None, False
    if budget:
        b1 = dict(body); b1["max_tokens"] = int(budget)
        r1, s1 = post(url + "/chat/completions", b1, timeout); secs += s1
        if "error" in r1:
            err = r1["error"]
        else:
            ch = r1["choices"][0]; msg = ch["message"]
            reasoning = msg.get("reasoning_content") or ""; content = msg.get("content") or ""; finish = ch.get("finish_reason")
            usage = r1.get("usage") or {}
            if finish == "length" and not content.strip():
                forced = True
                # continue the assistant turn through /v1/completions with the forced close of the think block
                eff = EFFORT_TEXT.get(arm.get("effort") or "xhigh", "")
                sysx = (eff + "\n\n" + prompt["system"]) if eff else prompt["system"]
                text = render_chatml(sysx, prompt["user"], "<think>\n" + reasoning + (arm.get("budget_text") or DEFAULT_BUDGET_TEXT))
                b2 = {"model": "qwen38-27b", "prompt": text, "max_tokens": max(256, max_tokens - int(budget)), "seed": seed}
                for k in ("presence_penalty", "temperature", "top_p", "steering"):
                    if k in body: b2[k] = body[k]
                r2, s2 = post(url + "/completions", b2, timeout); secs += s2
                if "error" in r2:
                    err = r2["error"]
                else:
                    content = r2["choices"][0].get("text") or ""; finish = r2["choices"][0].get("finish_reason")
                    u2 = r2.get("usage") or {}
                    usage = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": (usage.get("completion_tokens") or 0) + (u2.get("completion_tokens") or 0),
                             "first_leg": usage, "second_leg": u2}
    else:
        r, s = post(url + "/chat/completions", body, timeout); secs = s
        if "error" in r:
            err = r["error"]
        else:
            ch = r["choices"][0]; msg = ch["message"]
            reasoning = msg.get("reasoning_content") or ""; content = msg.get("content") or ""; finish = ch.get("finish_reason")
            usage = r.get("usage") or {}
    onset, cmax, ctail = loop_stats(reasoning)
    low = reasoning.lower()
    n_eq, n_wrong, bad = check_equations(reasoning + "\n" + content)
    n_sum, n_sum_wrong, bad_sum = check_equations(reasoning + "\n" + content, SUM_EQ)
    row.update({"error": err, "finish": finish, "forced_close": forced, "seconds": round(secs, 1),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "reasoning_chars": len(reasoning), "reasoning_words": len(reasoning.split()), "content_chars": len(content),
                "loop_onset_char": onset, "compress_max": cmax, "compress_tail": ctail, "repeat8": ngram_repeat(reasoning),
                "pivots": {p: low.count(p) for p in PIVOTS}, "equations": n_eq, "equations_wrong": n_wrong, "wrong_examples": bad,
                "sums": n_sum, "sums_wrong": n_sum_wrong, "sums_wrong_examples": bad_sum,
                "answer": parse_answer(content), "usage": usage, "reasoning_tail": reasoning[-600:], "content": content[:4000]})
    with out_lock:
        fh.write(json.dumps(row) + "\n"); fh.flush()
        a = row["answer"]
        print(f"{pname:18s} {arm['name']:14s} seed {seed} {row['finish']!s:7s} {row['completion_tokens']!s:>6s} tok {row['seconds']:7.1f}s "
              f"think {row['reasoning_chars']:>6d}ch loop@{onset} cmax {cmax} wait {row['pivots']['wait']:3d} "
              f"q {a['q_hex_ok']}/4 log {int(a['has_log'])} eq {n_eq}/{n_wrong}wrong {('ERR ' + err[:80]) if err else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--seed-base", type=int, default=20261108)
    ap.add_argument("--max-tokens", type=int, default=24576); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600); ap.add_argument("--only", default=""); ap.add_argument("--only-arms", default="")
    a = ap.parse_args()
    prompts = json.load(open(a.prompts)); arms = json.load(open(a.arms))
    if a.only: prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    if a.only_arms: arms = [x for x in arms if x["name"] in a.only_arms.split(",")]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows_path = Path(a.out) / "rows.jsonl"
    done = set()
    if rows_path.exists():   # resume: skip (prompt, arm, seed) already answered without error
        for line in open(rows_path):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["prompt"], r["arm"], r["seed"]))
            except Exception:
                pass
    fh = open(rows_path, "a"); lock = threading.Lock()
    jobs = [(pn, p, arm, a.seed_base + s) for arm in arms for pn, p in prompts.items() for s in range(a.seeds)
            if (pn, arm["name"], a.seed_base + s) not in done]
    print(f"{len(jobs)} requests, {a.concurrency} at a time -> {a.out}/rows.jsonl", flush=True)
    sem = threading.Semaphore(a.concurrency); threads = []
    def run(job):
        with sem:
            one(a, a.url, job[0], job[1], job[2], job[3], a.max_tokens, a.timeout, lock, fh)
    for j in jobs:
        t = threading.Thread(target=run, args=(j,), daemon=True); t.start(); threads.append(t); time.sleep(0.2)
    for t in threads: t.join()


if __name__ == "__main__":
    main()
