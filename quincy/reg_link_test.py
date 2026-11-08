#!/usr/bin/env python3
"""read/write link between two frozen models: can a small trained head carry a register value from one to the other as
one embedding row, with no words in between?

a sender sees "your input register: 7a1f". a head maps the sender's residuals at layer L over those digits to m embedding
rows. a receiver gets the rows at a slot in its prompt, and never sees the value, and has to write it. the trunks are frozen:
only the head trains, and the loss on the receiver's reply flows back through the receiver into the head. rows go in through
inputs_embeds, so no vocabulary table is rewritten. write-head idea: neuropedia's GRAFT-PLAN "little memory head". hidden-state
links and the controls below: skizy's BeeLLM (https://huggingface.co/schizophyllume/BeeLLM).

arms, on values no training batch ever used:
  on        the head's rows from a sender who saw the value
  text      the value as plain text at the slot: what the receiver can do at best
  calm      the head's rows from a sender whose register shows no value
  random    random rows of the same length
  empty     zero rows

PASS BAR, fixed before the first run (2026-11-08): on trained prompt forms, "on" gets >= 80% exact on held-out values, with
calm, random and empty each <= 1% exact. chance on 4 hex digits is 1 in 65,536, 6.25% per digit. the prompt form the head
never trained on is reported beside the bar, not in it: it says whether the rows work outside the sentences they learned.

  python reg_link_test.py --model DIR --out DIR --layer 8 --m 1 --steps 6000
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SLOT = "\x00"
SENDER = "your input register: {v}  (changed)\nwhat is on your input register?"
CALM = "your input register: none  (unchanged)\nwhat is on your input register?"
TRAIN_FORMS = [
    f"link: {SLOT}\nwrite the value on the link as hex digits, nothing else.",
    f"a process next to you sent this: {SLOT}. copy it to your output, nothing else.",
    f"the value from the other process is {SLOT}. what is it? answer with the hex digits only.",
]
NEW_FORM = f"your output register has to hold what this carries: {SLOT}\nwrite the output register now."
EOS = (248046, 248044)
HELD_SEED = 20261108       # fixed across runs and arms, so every run holds out the same values


def render(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def say(**r):
    r["at"] = round(time.time(), 1)
    print(json.dumps(r), flush=True)


class Head(nn.Module):
    """residuals over up to 4 value tokens -> m rows. each row's length is held at the embedding median."""

    def __init__(self, d, m, length, hidden=256, width=2048, max_tokens=4):
        super().__init__()
        self.m, self.d, self.length, self.max_tokens, self.hidden = m, d, length, max_tokens, hidden
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, hidden)
        self.net = nn.Sequential(nn.Linear(hidden * max_tokens, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, m * d))

    def forward(self, h):                       # (batch, tokens, d)
        x = self.down(self.norm(h))
        x = F.pad(x, (0, 0, 0, self.max_tokens - x.shape[1]))
        rows = self.net(x.flatten(1)).view(-1, self.m, self.d)
        return rows / rows.norm(dim=-1, keepdim=True) * self.length


class Rig:
    def __init__(self, path, layer, dtype):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.dev = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map={"": self.dev}).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        inner = self.model.model.language_model if hasattr(self.model.model, "language_model") else self.model.model
        self.layer = inner.layers[layer]
        self.E = self.model.get_input_embeddings().weight
        self.d = self.E.shape[1]
        self.length = self.E[:248000].float().norm(dim=1).median().item()
        self.dtype = self.E.dtype

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def sender_ids(self, text, value):
        """token ids of a sender prompt, and the positions of the value's tokens in it."""
        full = render(text)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        start = full.index(value)
        pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < start + len(value) and e > start]
        return enc["input_ids"], pos

    @torch.no_grad()
    def residuals(self, batch_ids, batch_pos):
        out = {}

        def grab(_m, _a, o):
            out["h"] = (o[0] if isinstance(o, tuple) else o).detach().float()

        hook = self.layer.register_forward_hook(grab)
        self.model(input_ids=torch.tensor(batch_ids, device=self.dev), use_cache=False)
        hook.remove()
        idx = torch.tensor(batch_pos, device=self.dev)[:, :, None].expand(-1, -1, self.d)
        return out["h"].gather(1, idx)

    def receiver_embeds(self, form, rows, tail_ids=None):
        """the form's prompt with rows at the slot, then any reply tokens. rows: (batch, m, d)."""
        before, after = render(form).split(SLOT)
        b = len(rows)
        pre = self.E[torch.tensor(self.ids(before), device=self.dev)]
        post = self.E[torch.tensor(self.ids(after), device=self.dev)]
        parts = [pre.expand(b, -1, -1), rows.to(self.dtype), post.expand(b, -1, -1)]
        if tail_ids is not None:
            parts.append(self.E[torch.tensor(tail_ids, device=self.dev)])
        return torch.cat(parts, 1)

    @torch.no_grad()
    def greedy(self, embeds, n=9):
        done = [None] * len(embeds)
        seqs = [[] for _ in range(len(embeds))]
        x = embeds
        for _ in range(n):
            nxt = self.model(inputs_embeds=x, use_cache=False).logits[:, -1].argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if done[i] is None:
                    if t in EOS:
                        done[i] = True
                    else:
                        seqs[i].append(t)
            x = torch.cat([x, self.E[nxt][:, None]], 1)
            if all(done):
                break
        return [self.tok.decode(s, skip_special_tokens=True).strip().lower() for s in seqs]


def value_tables(rig):
    """every 4-digit value, grouped so a batch has no padding: same token count in the sender and as a reply."""
    held_rng = random.Random(HELD_SEED)
    values = [f"{i:04x}" for i in range(1, 65536)]
    held_rng.shuffle(values)
    held, train = values[:6000], values[6000:]

    def group(vals):
        out = {}
        for v in vals:
            ids, pos = rig.sender_ids(SENDER.format(v=v), v)
            reply = rig.ids(v)
            out.setdefault((len(ids), len(pos), len(reply)), []).append((v, ids, pos, reply))
        return {k: x for k, x in out.items() if k[1] <= 4}

    return group(train), group(held)


def pick(groups, rng, size):
    keys = list(groups)
    key = rng.choices(keys, weights=[len(groups[k]) for k in keys])[0]
    return rng.sample(groups[key], min(size, len(groups[key])))


def digit_score(reply, value):
    return sum(a == b for a, b in zip(reply[:4], value)) / 4 if len(reply) == 4 else 0.0


def in_chunks(groups, total, seed):
    """a fixed sample of held-out values, in same-shape batches, in proportion to how common each shape is."""
    rng = random.Random(seed)
    n = sum(len(x) for x in groups.values())
    for k, rows in sorted(groups.items()):
        take = rng.sample(rows, min(len(rows), max(1, round(total * len(rows) / n))))
        for i in range(0, len(take), 64):
            yield take[i:i + 64]


def score_arm(rig, head, form, groups, total, arm, seed=1):
    exact, digits, n, samples = 0, 0.0, 0, []
    calm_ids, _ = rig.sender_ids(CALM, "none")
    calm_pos = [i for i, (s, e) in enumerate(rig.tok(render(CALM), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"])
                if s < render(CALM).index("none") + 4 and e > render(CALM).index("none")]
    g = torch.Generator(device=rig.dev).manual_seed(seed)
    for chunk in in_chunks(groups, total, seed):
        vals = [c[0] for c in chunk]
        b = len(chunk)
        if arm == "text":
            before, after = render(form).split(SLOT)
            replies = []
            for v in vals:           # the value's token count can differ with its neighbours, so one at a time
                ids = torch.tensor([rig.ids(before + v + after)], device=rig.dev)
                replies += rig.greedy(rig.E[ids])
        else:
            if arm == "on":
                rows = head(rig.residuals([c[1] for c in chunk], [c[2] for c in chunk]))
            elif arm == "calm":
                rows = head(rig.residuals([calm_ids] * b, [calm_pos] * b))
            elif arm == "random":
                rows = torch.randn(b, head.m, head.d, device=rig.dev, generator=g)
                rows = rows / rows.norm(dim=-1, keepdim=True) * head.length
            elif arm == "empty":
                rows = torch.zeros(b, head.m, head.d, device=rig.dev)
            replies = rig.greedy(rig.receiver_embeds(form, rows))
        for v, r in zip(vals, replies):
            exact += r == v
            digits += digit_score(r, v)
            n += 1
            if len(samples) < 6:
                samples.append([v, r])
    return {"exact": round(exact / n, 4), "digit": round(digits / n, 4), "n": n, "samples": samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=1, help="rows per value")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--final-n", type=int, default=512)
    ap.add_argument("--gpu-gib", type=float, default=0, help="hard cap on this process's card memory: keep it to the lease")
    a = ap.parse_args()
    if a.gpu_gib:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    rig = Rig(a.model, a.layer, a.dtype)
    train, held = value_tables(rig)
    say(event="ready", d=rig.d, length=round(rig.length, 3), train=sum(map(len, train.values())), held=sum(map(len, held.values())),
        vram_gib=round(torch.cuda.memory_allocated() / 2**30, 2), args=vars(a))
    head = Head(rig.d, a.m, rig.length).to(rig.dev)
    say(event="head", params=sum(p.numel() for p in head.parameters()))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0, running = time.time(), None
    for step in range(1, a.steps + 1):
        chunk = pick(train, rng, a.batch)
        form = rng.choice(TRAIN_FORMS)
        rows = head(rig.residuals([c[1] for c in chunk], [c[2] for c in chunk]))
        targets = torch.tensor([c[3] + [EOS[0]] for c in chunk], device=rig.dev)
        x = rig.receiver_embeds(form, rows, targets[:, :-1].tolist())
        logits = rig.model(inputs_embeds=x, use_cache=False).logits[:, -targets.shape[1]:].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % a.every == 0 or step == 1:
            with torch.no_grad():
                quick = score_arm(rig, head, TRAIN_FORMS[0], held, 128, "on")
            say(event="step", step=step, loss=round(running, 4), on_exact=quick["exact"], on_digit=quick["digit"], samples=quick["samples"][:3],
                s_per_step=round((time.time() - t0) / step, 3), vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            torch.save({"head": head.state_dict(), "args": vars(a), "step": step}, out / "head.pt")

    results = {"args": vars(a), "loss": running, "minutes": round((time.time() - t0) / 60, 1), "forms": {}}
    with torch.no_grad():
        for name, form in [("trained_0", TRAIN_FORMS[0]), ("trained_1", TRAIN_FORMS[1]), ("trained_2", TRAIN_FORMS[2]), ("new_form", NEW_FORM)]:
            results["forms"][name] = {arm: score_arm(rig, head, form, held, a.final_n, arm) for arm in ("on", "text", "calm", "random", "empty")}
            say(event="form", form=name, **{arm: results["forms"][name][arm]["exact"] for arm in results["forms"][name]})
    trained = [results["forms"][f"trained_{i}"] for i in range(3)]
    mean = lambda arm: sum(t[arm]["exact"] for t in trained) / 3
    results["on_exact_trained"] = round(mean("on"), 4)
    results["controls_exact_trained"] = {arm: round(mean(arm), 4) for arm in ("calm", "random", "empty")}
    results["text_exact_trained"] = round(mean("text"), 4)
    results["passes"] = mean("on") >= 0.8 and all(mean(arm) <= 0.01 for arm in ("calm", "random", "empty"))
    torch.save({"head": head.state_dict(), "args": vars(a), "step": a.steps}, out / "head.pt")
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    say(event="done", passes=results["passes"], on=results["on_exact_trained"], text=results["text_exact_trained"],
        controls=results["controls_exact_trained"], new_form=results["forms"]["new_form"]["on"]["exact"])


if __name__ == "__main__":
    main()
