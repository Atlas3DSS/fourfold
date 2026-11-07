"""preflight for a fourfold run on Flare: wait until every endpoint in the config
answers, then send each one a real tick-1 prompt and check the reply parses.

  python quincy/smoke.py config.quincy.toml              # wait up to 30 min, then one call per endpoint
  python quincy/smoke.py config.quincy.toml --wait 0     # don't wait, fail fast
  python quincy/smoke.py config.quincy.toml --full       # also fill every slot at once (4 per endpoint)

exit status 0 only if every endpoint is up and every call returns a reply with q tags.
"ok*" marks a reply whose values the sim would reject (e.g. a wrong-length hex value):
that is the model's behaviour, which the run is meant to measure, not a plumbing fault.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx  # noqa: E402

import sim  # noqa: E402


async def wait_up(urls: list[str], deadline: float, need: int) -> list[str]:
    async with httpx.AsyncClient(timeout=5) as c:
        while True:
            down = []
            for u in urls:
                try:
                    r = await c.get(u.rstrip("/") + "/models")
                    if r.status_code != 200:
                        down.append(u)
                except httpx.HTTPError:
                    down.append(u)
            if len(urls) - len(down) >= need or time.time() > deadline:
                return down
            print(f"waiting for {len(down)}/{len(urls)} endpoints: {' '.join(down)}", flush=True)
            await asyncio.sleep(20)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--wait", type=float, default=30, help="minutes to wait for endpoints")
    ap.add_argument("--full", action="store_true", help="fill every slot at once")
    ap.add_argument("--min-up", type=int, help="endpoints that must answer (default: all; "
                    "with routing = least_loaded a server that is down is just skipped)")
    args = ap.parse_args()
    cfg = sim.load_config(args.config)
    sim.set_width(cfg)
    urls = cfg["llm"]["base_url"] if isinstance(cfg["llm"]["base_url"], list) else [cfg["llm"]["base_url"]]

    need = args.min_up or len(urls)
    down = await wait_up(urls, time.time() + args.wait * 60, need)
    if len(urls) - len(down) < need:
        sys.exit(f"FAIL: only {len(urls) - len(down)}/{len(urls)} endpoints answer, need {need}: "
                 f"down {' '.join(down)}")
    print(f"{len(urls) - len(down)}/{len(urls)} endpoints answer /models"
          + (f" (down: {' '.join(down)})" if down else ""))
    up = len(urls) - len(down)

    per = int(cfg["llm"].get("endpoint_concurrency") or 1) if args.full else 1
    meta = {"config": cfg, "prompts": {k: (sim.ROOT / "prompts" / f"{k}.txt").read_text() for k in ("system", "tick")}}
    agent = sim.make_agent(cfg)
    turns = []
    for aid in range(up * per):                 # pinned: agent id % len(urls) picks the endpoint
        st = sim.AgentState(id=aid, seed=f"{random.getrandbits(32):08x}")
        inp = [sim.ZERO] * 4
        turns.append(sim.Turn(t=1, state=st, inputs=inp, changed=[False] * 4,
                              system=sim.system_prompt(meta, st.seed),
                              prompt=sim.tick_prompt(meta["prompts"]["tick"], 1, st, inp, [False] * 4,
                                                     cfg["log_lines_shown"])))
    t0 = time.perf_counter()
    reps = await asyncio.gather(*(sim.safe_act(agent, tn) for tn in turns))
    await agent.close()
    ok = True
    for tn, rep in zip(turns, reps):
        ep = (rep.usage or {}).get("endpoint", tn.state.id % len(urls))
        if rep.error and not rep.raw:
            ok = False
            print(f"agent {tn.state.id} ep {ep}: ERROR {rep.error}")
            continue
        _, lines, _, errs, _ = sim.parse_response(rep.raw, tn.state.outputs, "", cfg["protocol_word_limit"])
        u = rep.usage or {}
        # plumbing fails the smoke: an api error or a reply with no q tags at all. a value of the
        # wrong length is the model's own slip; the sim rejects it and tells the agent, as designed
        missing = sum(e.endswith(": missing") for e in errs if e.startswith("q"))
        good = not rep.error and missing < 4
        ok &= good
        tag_ = "ok " if good and not errs else ("ok*" if good else "BAD")
        print(f"agent {tn.state.id} ep {ep}: {tag_} {rep.latency:6.1f}s  "
              f"prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')}  "
              f"reasoning {len(rep.reasoning or '')} chars  errors {errs + ([rep.error] if rep.error else [])}")
    print(f"{len(turns)} calls in {time.perf_counter() - t0:.1f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
