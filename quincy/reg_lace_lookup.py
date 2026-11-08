#!/usr/bin/env python3
"""can a steering direction, measured on the 0.8B itself, make it better at looking up a value by address?

prompt: a 16-entry board of address:value pairs, then "what is the value at address xy?". the reply is held to hex digits by a
mask on the logits. on the running server the 0.8B got 26/60 this way (chance 1 in 16), and on small boards it often answers with
the address instead of the value.

direction: run many prompts, split them into right and wrong answers, and take the difference of mean residuals at the last
prompt token, per layer. the difference is taken within each target cell and then averaged, so it cannot just learn which cells
are easy. the push adds alpha x (median residual length) x unit direction to that layer's residual, at every position.

fresh random boards at each stage: fit (measure), val (pick layer and alpha, + direction only), test (read once).

PASS BAR, fixed before the first run (2026-11-08): on the test boards the lace beats no lace by at least 10 points exact, and each
of 3 random directions of the same size gains less than half of what the lace gains. the opposite direction and a 4-entry board
are reported beside the bar. this is an exploratory look on one model, so a pass means "worth pursuing", not "solved".

  python reg_lace_lookup.py --model DIR --out results.json
"""
import argparse
import json
import random
import re
import time

import torch

LAYERS = (3, 7, 11, 15, 19, 23)
ALPHAS = (0.05, 0.1, 0.2, 0.4)


def render(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def say(**r):
    print(json.dumps(r), flush=True)


def make_case(rng, rows, cols):
    board = {f"{r:x}{c:x}": f"{rng.getrandbits(8):02x}" for r in range(rows) for c in range(cols)}
    addr = rng.choice(list(board))
    lines = "\n".join("  ".join(f"{r:x}{c:x}:{board[f'{r:x}{c:x}']}" for c in range(cols)) for r in range(rows))
    return dict(prompt=f"{lines}\nwhat is the value at address {addr}? answer with the two hex digits only.", addr=addr,
                gold=board[addr], board=board)


class Rig:
    def __init__(self, path, dtype, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.dev = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        inner = self.model.model.language_model if hasattr(self.model.model, "language_model") else self.model.model
        self.layers = inner.layers
        vocab = self.tok.get_vocab()
        self.one = torch.tensor([i for t, i in vocab.items() if re.fullmatch(r"[0-9a-f]", t)], device=self.dev)
        self.two = torch.tensor([i for t, i in vocab.items() if re.fullmatch(r"[0-9a-f]{2}", t)], device=self.dev)

    def batches(self, cases, size):
        """same-length prompts only, so nothing is padded."""
        groups = {}
        for c in cases:
            c["ids"] = self.tok(render(c["prompt"]), add_special_tokens=False)["input_ids"]
            groups.setdefault(len(c["ids"]), []).append(c)
        for rows in groups.values():
            for i in range(0, len(rows), size):
                yield rows[i:i + size]

    @torch.no_grad()
    def forward(self, ids, push=None, capture=None):
        handles = []
        if push:
            layer, vec = push

            def add(_m, _a, o):
                if isinstance(o, tuple):
                    return (o[0] + vec.to(o[0].dtype), *o[1:])
                return o + vec.to(o.dtype)

            handles.append(self.layers[layer].register_forward_hook(add))
        if capture is not None:
            for L in LAYERS:
                def grab(_m, _a, o, L=L):
                    capture[L] = (o[0] if isinstance(o, tuple) else o)[:, -1].detach().float().cpu()

                handles.append(self.layers[L].register_forward_hook(grab))
        try:
            return self.model(input_ids=ids, use_cache=False).logits[:, -1]
        finally:
            for h in handles:
                h.remove()

    def answer(self, cases, push=None, capture_into=None, size=32):
        """masked greedy: a two-digit token ends it, a one-digit token is followed by one more digit."""
        for chunk in self.batches(cases, size):
            ids = torch.tensor([c["ids"] for c in chunk], device=self.dev)
            cap = {} if capture_into is not None else None
            logits = self.forward(ids, push, cap)
            if cap is not None:
                for L, h in cap.items():
                    capture_into.setdefault(L, []).append(h)
            allowed = torch.cat([self.one, self.two])
            first = allowed[logits[:, allowed].argmax(-1)]
            text = [self.tok.decode([t]) for t in first.tolist()]
            ids2 = torch.cat([ids, first[:, None]], 1)
            logits2 = self.forward(ids2, push)
            second = self.one[logits2[:, self.one].argmax(-1)]
            for c, t, s in zip(chunk, text, second.tolist()):
                c["reply"] = t if len(t) == 2 else t + self.tok.decode([s])


def tally(cases):
    n = len(cases)
    right = sum(c["reply"] == c["gold"] for c in cases)
    address = sum(c["reply"] == c["addr"] and c["reply"] != c["gold"] for c in cases)
    other_entry = sum(c["reply"] != c["gold"] and c["reply"] != c["addr"] and c["reply"] in c["board"].values() for c in cases)
    return {"n": n, "exact": round(right / n, 4), "said_the_address": round(address / n, 4),
            "said_another_entry": round(other_entry / n, 4), "said_something_else": round((n - right - address - other_entry) / n, 4)}


def directions(cases, cap):
    """per layer: within-cell mean(right) - mean(wrong), averaged over cells, as a unit vector, and the median residual length."""
    out = {}
    order = [c for chunk_cases in [cases] for c in chunk_cases]
    for L, chunks in cap.items():
        h = torch.cat(chunks)
        cells = {}
        for row, c in zip(h, order):
            cells.setdefault(c["addr"], {"r": [], "w": []})["r" if c["reply"] == c["gold"] else "w"].append(row)
        diffs = [torch.stack(v["r"]).mean(0) - torch.stack(v["w"]).mean(0) for v in cells.values() if v["r"] and v["w"]]
        d = torch.stack(diffs).mean(0)
        out[L] = (d / d.norm(), h.norm(dim=1).median().item(), len(diffs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--fit", type=int, default=3000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--gpu-gib", type=float, default=0)
    a = ap.parse_args()
    if a.gpu_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    rig = Rig(a.model, a.dtype, a.device)
    say(event="ready", one=len(rig.one), two=len(rig.two))

    def cases(seed, n, rows=4, cols=4):
        rng = random.Random(seed)
        return [make_case(rng, rows, cols) for _ in range(n)]

    fit = cases("fit", a.fit)
    groups = list(rig.batches(fit, a.size))
    ordered = [c for g in groups for c in g]           # capture order: the same batches answer() will walk
    cap = {}
    rig.answer(fit, capture_into=cap, size=a.size)
    base_fit = tally(fit)
    say(event="fit", **base_fit)
    dirs = directions(ordered, cap)
    say(event="directions", cells={L: dirs[L][2] for L in dirs}, lengths={L: round(dirs[L][1], 2) for L in dirs})

    def score(seed, n, push_spec=None, rows=4, cols=4):
        cs = cases(seed, n, rows, cols)
        push = None
        if push_spec:
            layer, vec = push_spec
            push = (layer, vec.to(rig.dev))
        rig.answer(cs, push=push, size=a.size)
        return tally(cs)

    def vec(L, alpha, sign=1.0):
        d, length, _ = dirs[L]
        return sign * alpha * length * d

    val_base = score("val", a.val)
    say(event="val_base", **val_base)
    best = None
    for L in LAYERS:
        for alpha in ALPHAS:
            r = score("val", a.val, (L, vec(L, alpha)))
            say(event="val", layer=L, alpha=alpha, **r)
            if best is None or r["exact"] > best[2]["exact"]:
                best = (L, alpha, r)
    L, alpha = best[0], best[1]
    say(event="picked", layer=L, alpha=alpha, val=best[2]["exact"], val_base=val_base["exact"])

    results = {"args": vars(a), "fit_base": base_fit, "val_base": val_base, "picked": {"layer": L, "alpha": alpha, "val": best[2]}}
    results["test_base"] = score("test", a.test)
    results["test_lace"] = score("test", a.test, (L, vec(L, alpha)))
    results["test_opposite"] = score("test", a.test, (L, vec(L, alpha, -1.0)))
    randoms = []
    for s in range(3):
        g = torch.Generator().manual_seed(s)
        r = torch.randn(rig.model.get_input_embeddings().weight.shape[1], generator=g)
        randoms.append(score("test", a.test, (L, alpha * dirs[L][1] * r / r.norm())))
    results["test_random"] = randoms
    results["four_base"] = score("four", a.test, rows=2, cols=2)
    results["four_lace"] = score("four", a.test, (L, vec(L, alpha)), rows=2, cols=2)
    gain = results["test_lace"]["exact"] - results["test_base"]["exact"]
    random_gains = [r["exact"] - results["test_base"]["exact"] for r in randoms]
    results["gain"] = round(gain, 4)
    results["random_gains"] = [round(g, 4) for g in random_gains]
    results["passes"] = gain >= 0.10 and all(g < gain / 2 for g in random_gains)
    results["minutes"] = round((time.time() - t0) / 60, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    say(event="done", passes=results["passes"], gain=results["gain"], random_gains=results["random_gains"], base=results["test_base"],
        lace=results["test_lace"], opposite=results["test_opposite"]["exact"], four_base=results["four_base"]["exact"],
        four_lace=results["four_lace"]["exact"], minutes=results["minutes"])


if __name__ == "__main__":
    main()
