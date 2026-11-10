"""the tick floor of each method with perfect scripted agents, through the real sim (bridge, typed writes, seams).

  .venv/bin/python experiments/tree_ring/floor.py            # the table
  .venv/bin/python experiments/tree_ring/floor.py --miss 0.05 --trials 20   # robustness: agents randomly give no reply

no model, no gpu, seconds. worlds use seed 20261108, the same as the llm runs.
"""
import argparse
import asyncio
import copy
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sim  # noqa: E402

WORLDS = [   # name, rows, cols, topology, extra config, methods
    ("4x4 torus", 4, 4, "torus", {}, ("flood", "tree", "tree2")),
    ("8x8 torus", 8, 8, "torus", {}, ("flood", "tree", "tree2")),
    ("10x10 torus", 10, 10, "torus", {}, ("flood", "tree", "tree2")),
    ("4x28 torus, 7 arrays, seams every 2", 4, 28, "torus", {"modules": [4, 4], "seam_period": 2}, ("flood", "tree", "tree2")),
    ("20x20 torus", 20, 20, "torus", {}, ("tree", "tree2")),
    ("tesseract side 3", 9, 9, "tesseract", {"side": 3}, ("flood", "ring")),
    ("tesseract side 4", 16, 16, "tesseract", {"side": 4}, ("flood", "ring")),
    ("tesseract side 5", 25, 25, "tesseract", {"side": 5}, ("ring",)),
]


def far_root(rows, cols):
    return (rows // 2) * cols + cols // 2


async def one(rows, cols, topo, extra, method, miss=0.0, seed=20261108, cap=400, tmp=None):
    cfg = copy.deepcopy(sim.DEFAULTS)
    cfg.update(agent="scripted", rows=rows, cols=cols, topology=topo, world_seed=seed, actions=True, task="sum", say_only=True,
               structure_hint=True, notes_max=max(160, rows * cols + 8), store_prompts=False, **extra)
    cfg["scripted"] = {"kind": "tree" if method == "tree2" else method, "miss_rate": miss}
    if method in ("tree", "tree2"):
        cfg["protocol_hint"] = "tree"
        cfg["tree_roots"] = [0] if method == "tree" else [0, far_root(rows, cols)]
    elif method == "ring":
        cfg["protocol_hint"] = "ring"
    sim.set_width(cfg)
    sim.RUNS = Path(tmp)
    _, meta = sim.new_run(cfg, f"{method}-{rows}x{cols}-{topo}-{miss}-{seed}")
    states = sim.restore_states(meta, [])
    agent = sim.make_agent(cfg)
    n, ans = len(meta["agents"]), meta["answer"]
    want = {f"total {ans} over {n} processes", f"sum so far {ans} over {n} processes"}
    first = done = None
    wrong, rejected, tokens = set(), 0, []
    for t in range(1, cap + 1):
        rec = await sim.step(t, meta, states, agent)
        exact = 0
        for a in rec["agents"]:
            p = a["protocol"].replace(" (trees A and B agree)", "")
            exact += p in want
            if p.startswith("total ") and p not in want:
                wrong.add(a["agent"])
            rejected += sum("rejected" in x for x in a.get("actions", []))
        if exact and first is None:
            first = t
        if exact == n:
            done = t
            break
    chars = [len(sim.tick_prompt(meta["prompts"]["tick"], t, st, st.prev_inputs, [False] * 4, 20)
                 + sim.bridge_state_block(st, True)) for st in states]
    return {"n": n, "first": first, "all": done, "wrong": len(wrong), "rejected": rejected,
            "exact_at_cap": exact, "tick_chars": int(statistics.median(chars))}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows_out = []
    with tempfile.TemporaryDirectory() as tmp:
        print(f"{'world':<38} {'agents':>6} {'method':>6} {'first exact':>11} {'all exact':>9} {'wrong':>5} {'rejected':>8} {'tick prompt chars':>17}")
        for name, r, c, topo, extra, methods in WORLDS:
            if args.only and args.only not in name:
                continue
            for m in methods:
                if args.trials == 1:
                    x = await one(r, c, topo, extra, m, args.miss, tmp=tmp)
                    print(f"{name:<38} {x['n']:>6} {m:>6} {str(x['first']):>11} {str(x['all']):>9} {x['wrong']:>5} {x['rejected']:>8} {x['tick_chars']:>17}", flush=True)
                    rows_out.append({"world": name, "method": m, **x})
                else:
                    xs = [await one(r, c, topo, extra, m, args.miss, seed=20261108 + k, tmp=tmp) for k in range(args.trials)]
                    solved = [x["all"] for x in xs if x["all"]]
                    print(f"{name:<38} {xs[0]['n']:>6} {m:>6}  miss {args.miss}: solved {len(solved)}/{len(xs)}, "
                          f"all-exact tick median {statistics.median(solved) if solved else '-'} max {max(solved) if solved else '-'}, "
                          f"runs with a wrong total {sum(bool(x['wrong']) for x in xs)}", flush=True)
                    rows_out.append({"world": name, "method": m, "miss": args.miss, "trials": xs})
    if args.json:
        Path(args.json).write_text(json.dumps(rows_out, indent=1))


asyncio.run(main())
