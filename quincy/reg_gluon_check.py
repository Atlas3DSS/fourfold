#!/usr/bin/env python3
"""before a real run: can a real gluon carry out each kind of instruction, through the same test the host uses?

each case is one instruction at a position that allows it, scored by registers.score_unit_reply on fresh values, the bar is unit_bar.
needs the pool's server up. no run is created.

  python quincy/reg_gluon_check.py --config config.registers-two-arrays.toml --pool 0.8b --trials 12
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import registers  # noqa: E402
import sim  # noqa: E402

CASES = [("11", "copy p0 to q1"), ("11", "copy p2 to q3"), ("11", "copy p0 to q1; copy p2 to q3"), ("11", "copy p1 to private"),
         ("11", "copy private to q0"), ("11", "copy public to public"), ("11", "copy private.hi to public"), ("11", "copy private.lo to public"),
         ("11", "copy public to private.hi"), ("11", "copy public to private.lo"), ("00", "copy input to q2"), ("00", "copy input.hi to public"),
         ("00", "copy input.lo to public"), ("07", "copy p3 to out"), ("07", "copy private to out")]


async def check(cfg, agent, trials, pool=""):
    sim.set_width(cfg)
    rc = cfg["registers"]
    port_hex = int(cfg.get("port_bits", 128)) // 4
    meta = {"run_id": "gluon-check", "world_seed": 1, "rows": cfg["rows"], "cols": cfg["cols"], "topology": cfg.get("topology", "grid"), "config": cfg,
            "agents": sim.build_world(cfg["rows"], cfg["cols"], random.Random(1), torus=cfg.get("topology") == "torus")}
    registers.BANKS.pop("gluon-check", None)
    bank = registers.bank_for(meta)
    at = {registers.address(a): a["id"] for a in meta["agents"]}
    rng = random.Random(20261108)
    draw = lambda n: f"{rng.randrange(1, 16 ** n):0{n}x}"
    out = []
    for cell, program in CASES:
        new, why = registers.check_program(program, rc, port_hex)
        assert new, f"{program}: {why}"
        i = at[cell]
        cases = []
        for _ in range(trials):
            have = {f"p{k}": draw(port_hex) for k in range(4)}
            have.update(public=draw(int(rc["public_hex"])), private=draw(int(rc["private_hex"])), input=draw(int(rc["input_hex"])))
            cases.append(have)
        turns = [sim.Turn(t=1, state=sim.AgentState(id=i, seed="00000000", pool=pool), inputs=[h[f"p{k}"] for k in range(4)], changed=[False] * 4,
                          system=registers.gluon_system(cfg),
                          prompt=registers.gluon_tick(meta, bank, i, 1, [h[f"p{k}"] for k in range(4)], ["0" * port_hex] * 4, program=new,
                                                      public=h["public"], private=h["private"], input_value=h["input"]),
                          extra_body=registers.gluon_grammar(meta, i, port_hex, new)) for h in cases]
        replies = await asyncio.gather(*(sim.safe_act(agent, t) for t in turns))
        scores = [registers.score_unit_reply(r.raw, registers.expected_outputs(new, h), cell, rc, port_hex, h) for r, h in zip(replies, cases)]
        mean = sum(scores) / len(scores)
        out.append({"cell": cell, "program": new, "score": round(mean, 3), "exact": sum(s >= 0.999 for s in scores), "trials": trials,
                    "passes": mean >= float(rc["unit_bar"]), "sample": (replies[0].raw or "")[:120]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", default="0.8b")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = sim.load_config(a.config)
    p = next(x for x in cfg["llm"]["pools"] if x["name"] == a.pool)
    agent = sim.LLMAgent(sim.pool_config(cfg["llm"], p))
    rows = asyncio.run(check(cfg, agent, a.trials, a.pool))
    asyncio.run(agent.close())
    for r in rows:
        print(f"{'PASS' if r['passes'] else 'FAIL'}  {r['score']:6.3f}  {r['exact']:2d}/{r['trials']}  {r['cell']}  {r['program']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
