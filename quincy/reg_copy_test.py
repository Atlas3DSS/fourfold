#!/usr/bin/env python3
"""can each model size keep the register format and copy a value? the gate before any mixed-size array.

each pool gets the real system prompt and the real tick prompt, plus one plain instruction for that tick:
  copy   set your public register to the value at address xy (15 other values on the board as distractors)
  input  at the input position: public = the first two digits of the input, private = all of it
and every trial checks that the reply has a full format. in copy, the private register must also stay as it was.
chance on a copy is 1 in 256; on the 4-digit input it is 1 in 65,536.

proposed bar, per pool: format_ok >= 0.9, copy_exact >= 0.8, input_exact >= 0.8. this is not an owner rule.

  python quincy/reg_copy_test.py --config config.registers-4x4.toml --trials 30 --out results.json
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import registers  # noqa: E402
import sim  # noqa: E402

BAR = {"format_ok": 0.9, "copy_exact": 0.8, "input_exact": 0.8}


def _meta(cfg: dict, seed) -> dict:
    meta = {
        "run_id": f"copytest-{seed}",
        "world_seed": seed,
        "rows": cfg["rows"],
        "cols": cfg["cols"],
        "topology": cfg.get("topology", "grid"),
        "config": cfg,
        "prompts": {
            "system": (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8"),
            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
        },
        "agents": sim.build_world(cfg["rows"], cfg["cols"], random.Random(seed), torus=cfg.get("topology") == "torus"),
    }
    registers.BANKS.pop(meta["run_id"], None)
    return meta


def build_trial(cfg: dict, kind: str, trial: int) -> dict:
    """a world, a board and one instruction. drawn from the trial number alone, so every pool gets the same trials."""
    rng = random.Random(f"copytest:{kind}:{trial}")
    meta = _meta(cfg, rng.getrandbits(32))
    bank = registers.bank_for(meta)
    n = len(meta["agents"])
    pub_n, priv_n = int(cfg["registers"]["public_hex"]), int(cfg["registers"]["private_hex"])
    bank.pub = [f"{rng.getrandbits(pub_n * 4):0{pub_n}x}" for _ in range(n)]
    bank.priv = [f"{rng.getrandbits(priv_n * 4):0{priv_n}x}" for _ in range(n)]
    want: dict = {}
    if kind == "copy":
        i = rng.randrange(n)
        j = rng.choice([x for x in range(n) if x != i and bank.pub[x] != bank.pub[i]])
        addr = registers.address(meta["agents"][j])
        note = (f"instruction for this tick: set your public register to the value of the public register at address "
                f"{addr}. keep your private register as it is.")
        want = {"pub": bank.pub[j], "priv": bank.priv[i]}
    else:
        cell = str(cfg["registers"]["input_cell"]).lower()
        i = next(a["id"] for a in meta["agents"] if registers.address(a) == cell)
        _, value = registers.value_at(meta, bank, 1)
        note = ("instruction for this tick: set your public register to the first two digits of the input register, "
                "and your private register to the whole input register.")
        want = {"pub": value[:pub_n], "priv": value}
    a = meta["agents"][i]
    st = sim.AgentState(id=i, seed=a["seed"])
    inputs = ["0" * sim.HEXN] * 4
    prompt = (sim.tick_prompt(meta["prompts"]["tick"], 1, st, inputs, [False] * 4, 20)
              + registers.tick_block(meta, bank, i, 1) + "\n" + note + "\n")
    turn = sim.Turn(t=1, state=st, inputs=inputs, changed=[False] * 4, system=sim.system_prompt(meta, a["seed"]), prompt=prompt)
    return {"kind": kind, "trial": trial, "turn": turn, "addr": registers.address(a), "want": want, "cfg": cfg}


def score(trial: dict, reply) -> dict:
    cfg = trial["cfg"]
    rc = cfg["registers"]
    raw = reply.raw or ""
    pub, priv, first, errs, _ = registers.parse_reg(raw, trial["addr"], int(rc["public_hex"]), int(rc["private_hex"]))
    st = trial["turn"].state
    _, _, _, perr, _ = sim.parse_response(raw, st.outputs, st.protocol, cfg["protocol_word_limit"])
    row = {
        "kind": trial["kind"],
        "trial": trial["trial"],
        "format_ok": bool(raw) and not errs and not perr,
        "led_first": first,
        "exact": pub == trial["want"]["pub"] and (trial["kind"] == "copy" or priv == trial["want"]["priv"]),
        "hold_private": priv == trial["want"]["priv"] if trial["kind"] == "copy" else None,
        "error": reply.error,
    }
    if not row["exact"] or not row["format_ok"]:
        row["want"] = trial["want"]
        row["got"] = {"pub": pub, "priv": priv}
        row["raw"] = raw[-400:]
    return row


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 3) if rows else 0.0


def report(rows: list[dict]) -> dict:
    copies = [r for r in rows if r["kind"] == "copy"]
    inputs = [r for r in rows if r["kind"] == "input"]
    out = {
        "trials": len(rows),
        "format_ok": _rate(rows, "format_ok"),
        "led_first": _rate(rows, "led_first"),
        "copy_exact": _rate(copies, "exact"),
        "input_exact": _rate(inputs, "exact"),
        "hold_private": _rate(copies, "hold_private"),
        "failures": [r for r in rows if not r["exact"] or not r["format_ok"]][:8],
    }
    out["passes"] = all(out[k] >= v for k, v in BAR.items())
    return out


class Scripted:
    """a stand-in model for testing this script: follows the instruction, garbling the value at rate `garble`."""

    def __init__(self, garble: float = 0.0):
        self.garble = garble

    async def act(self, turn):
        st, p = turn.state, turn.prompt
        rng = random.Random(f"{st.seed}:{p}")
        addr = re.search(r"your address: (\w\w)", p).group(1)
        priv = re.search(r"your private register: (\w+)", p).group(1)
        m = re.search(r"public register at address (\w\w)", p)
        if m:
            pub = dict(re.findall(r"\b([0-9a-f]{2}):([0-9a-f]{2})\b", p))[m.group(1)]
        else:
            priv = re.search(r"input register at your position: (\w+)", p).group(1)
            pub = priv[:2]
        if rng.random() < self.garble:
            pub = f"{int(pub, 16) ^ 0xff:02x}"
            priv = f"{int(priv, 16) ^ 0xffff:04x}"
        body = "".join(f"<q{i}>{v}</q{i}>\n" for i, v in enumerate(st.outputs))
        return sim.Reply(raw=f"<reg>{addr} {pub} {priv}</reg>\n<log>ok</log>\n{body}")

    async def close(self):
        pass


def pool_agent(cfg: dict, name: str):
    for p in cfg["llm"].get("pools") or []:
        if p["name"] == name:
            return sim.LLMAgent(sim.pool_config(cfg["llm"], p))
    return sim.LLMAgent({k: v for k, v in cfg["llm"].items() if k != "pools"})


async def run_all(cfg: dict, trials: int, make=None, only=None) -> dict:
    cfg = copy.deepcopy(cfg)
    sim.set_width(cfg)
    make = make or (lambda name: pool_agent(cfg, name))
    names = [p["name"] for p in (cfg["llm"].get("pools") or [])] or ["all"]
    if only:
        names = [x for x in names if x in only]
    built = [build_trial(cfg, kind, i) for kind in ("copy", "input") for i in range(trials)]
    out = {}
    for name in names:
        agent = make(name)
        try:
            replies = await asyncio.gather(*(sim.safe_act(agent, t["turn"]) for t in built))
        finally:
            await agent.close()
        out[name] = report([score(t, r) for t, r in zip(built, replies)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=30, help="trials of each kind, per pool")
    ap.add_argument("--out", help="write the full report here as json")
    ap.add_argument("--pools", default="", help="comma list: only these pools")
    ap.add_argument("--set", action="append", default=[], metavar="POOL:JSON",
                    help="""settings over one pool's, e.g. 4b:{"max_tokens":4096}""")
    args = ap.parse_args()
    sim.load_dotenv()
    cfg = sim.load_config(args.config)
    registers.validate(cfg)
    for item in args.set:
        name, _, raw = item.partition(":")
        next(p for p in cfg["llm"]["pools"] if p["name"] == name).update(json.loads(raw))
    out = asyncio.run(run_all(cfg, args.trials, only=[x for x in args.pools.split(",") if x]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, r in out.items():
        print(f"{name:8s} format {r['format_ok']:.2f}  led first {r['led_first']:.2f}  copy {r['copy_exact']:.2f}  "
              f"input {r['input_exact']:.2f}  hold private {r['hold_private']:.2f}  {'PASS' if r['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
