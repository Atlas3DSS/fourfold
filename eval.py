"""ask every agent the final questions and score the answers against the true wiring.

  python eval.py                         # newest run, at its last tick
  python eval.py <run> --tick 30         # as if the run had ended after tick 30
  python eval.py <run> --show            # also print each agent's free-text answers

the agent gets its usual system prompt plus prompts/eval.txt: its final ports,
protocol and recent log, then the questions. it is called with the run's own
model and settings. this only reads ticks.jsonl, so it is safe on a live run;
results go to runs/<run>/eval-t<tick>.json.

scoring (all by code, no llm judge):
  count        1 - |claimed - true| / true, floored at 0
  arrangement  1 for a grid with the right dimensions, 0.5 for a grid, else 0
  position     the agent's role (corner / edge / centre, which is all an agent can
               know without a compass) against the truth, by keyword
  map          a drawn grid in the arrangement answer, matched to the truth under the 8
               rotations/reflections and any offset: (ids placed right - ids placed
               wrong) / agents, floored at 0. no drawn map scores 0
  wiring       per port slot (p0-p3, q0-q3): a dead slot scores 1 if called dead;
               a live slot scores 0.5 for the right neighbour id and 0.5 more
               for the right port on that neighbour
"""
from __future__ import annotations

import argparse
import os
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import sim
from runlog import resolve

ROOT = Path(__file__).resolve().parent
ID = re.compile(r"\b[0-9a-f]{8}\b")
DEAD_WORDS = re.compile(r"\b(nothing|none|nowhere|dead|zero|zeros|unconnected|disconnected|no (?:source|destination|connection|link|one))\b")


def tag(text: str, name: str) -> str | None:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text or "", re.S | re.I)
    return m[-1].strip() if m else None


def role_of(n_live: int, total_live_max: int = 4) -> str:
    return {4: "centre", 3: "edge", 2: "corner"}.get(n_live, f"{n_live}-link")


def claimed_role(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    # "upper-middle", "edge-middle", "middle of a side" describe an edge, not the centre
    t = re.sub(r"\b(upper|top|lower|bottom|left|right|edge|side|border)[- ]middle\b", "edge", t)
    t = re.sub(r"\bmiddle of (?:a|an|one|the \w+|its) (?:side|edge|row|column|border)\b", "edge", t)
    hits = [(m.start(), r) for r, pat in (("centre", r"\b(cent(?:er|re|ral)|middle)\b"),
                                          ("corner", r"\bcorner"),
                                          ("edge", r"\b(edge|side|border|perimeter)\b"))
            for m in [re.search(pat, t)] if m]
    return min(hits)[1] if hits else None


WRAP_WORDS = re.compile(r"\b(torus|toroid\w*|donut|doughnut|wrap\w*|periodic|cyclic|circular)\b|"
                        r"\bno (?:edges?|boundar\w*|borders?|corners?)\b")


def score_arrangement(text: str | None, rows: int, cols: int, torus: bool = False) -> float:
    """grid: 1 for the right dimensions, 0.5 for a grid. torus: 0.5 for the right
    dimensions plus 0.5 for saying the edges wrap round."""
    if not text:
        return 0.0
    t = text.lower()
    wraps = bool(WRAP_WORDS.search(t))
    if not re.search(r"\b(grid|lattice|mesh|array|matrix)\b", t) and not (torus and wraps):
        return 0.0
    dims = {f"{rows}x{cols}", f"{cols}x{rows}"}
    found = {f"{a}x{b}" for a, b in re.findall(r"(\d+)\s*(?:x|×|by)\s*(\d+)", t)}
    words = {"three by three"} if (rows, cols) == (3, 3) else set()
    right = bool(found & dims or any(w in t for w in words))
    if torus:
        return 0.5 * right + 0.5 * wraps
    return 1.0 if right else 0.5


MAP_TOKEN = re.compile(r"(?<![0-9a-z])([0-9a-f]{8}|unknown|empty|none|\?{1,3})(?![0-9a-z])")
MAP_GLUE = re.compile(r"^[\s—–\-|·.:+=<>…*~_()\[\]]*$")   # what may sit between cells in a drawn row


def parse_map(text: str | None) -> dict | None:
    """first drawn grid in the text -> {(row, col): id or None}. a row is a line
    with 2+ cells (ids or placeholders) and nothing else but connector symbols.
    columns are assigned by character position against the widest row, so rows
    with a missing cell still line up."""
    if not text:
        return None
    block = []
    for line in text.lower().splitlines():
        toks = [(m.start(), m.end(), m.group(1)) for m in MAP_TOKEN.finditer(line)]
        if len(toks) >= 2 and MAP_GLUE.match(MAP_TOKEN.sub("", line)) and \
                (any(ID.fullmatch(t[2]) for t in toks) or block):
            block.append(toks)
        elif not line.strip() or MAP_GLUE.match(line):
            continue                      # blank or "|   |   |" connector lines
        elif block:
            break                         # prose after the grid ends it
    if len(block) < 2:
        return None
    ref = max(block, key=len)
    centres = [(s + e) / 2 for s, e, _ in ref]
    cells = {}
    for r, toks in enumerate(block):
        for s, e, tok in toks:
            c = min(range(len(centres)), key=lambda i: abs(centres[i] - (s + e) / 2))
            cells.setdefault((r, c), tok if ID.fullmatch(tok) else None)
    return cells


def score_map(cells: dict | None, meta: dict) -> tuple[float, int, int]:
    """best match of a drawn map against the true grid over the 8 rotations/
    reflections and every offset that fits (on a torus every cyclic shift, since
    no cell is special). -> (score, correct, wrong) where
    score = max(0, (correct - wrong) / n_agents)."""
    if not cells:
        return 0.0, 0, 0
    R, C = meta["rows"], meta["cols"]
    truth = {tuple(a["pos"]): a["seed"] for a in meta["agents"]}
    torus = meta.get("topology") == "torus"
    best = (0.0, 0, 0)
    for swap in (False, True):
        for fr in (1, -1):
            for fc in (1, -1):
                pts = {}
                for (r, c), v in cells.items():
                    rr, cc = (c, r) if swap else (r, c)
                    pts[(fr * rr, fc * cc)] = v
                r0, c0 = min(p[0] for p in pts), min(p[1] for p in pts)
                pts = {(r - r0, c - c0): v for (r, c), v in pts.items()}
                h, w = max(p[0] for p in pts) + 1, max(p[1] for p in pts) + 1
                if torus and (h > R or w > C):
                    continue
                at = (lambda r, c: ((r % R), (c % C))) if torus else (lambda r, c: (r, c))
                for dr in range(R if torus else R - h + 1):
                    for dc in range(C if torus else C - w + 1):
                        ok = sum(1 for (r, c), v in pts.items() if v and truth.get(at(r + dr, c + dc)) == v)
                        bad = sum(1 for (r, c), v in pts.items() if v and truth.get(at(r + dr, c + dc)) != v)
                        s = max(0.0, (ok - bad) / len(truth))
                        if (s, ok) > (best[0], best[1]):
                            best = (s, ok, bad)
    return best


def parse_slot(text: str | None, own_id: str, want_letter: str):
    """-> ("dead", None, None) | ("live", id, port or None) | ("none", None, None)"""
    if not text:
        return "none", None, None
    t = text.lower()
    ids = [m for m in ID.finditer(t) if m.group(0) != own_id and m.group(0) != "00000000"]
    if ids:
        m = ids[0]
        pm = re.search(rf"\b{want_letter}\s*#?\s*([0-3])\b", t[m.end():])
        return "live", m.group(0), int(pm.group(1)) if pm else None
    if DEAD_WORDS.search(t):
        return "dead", None, None
    return "none", None, None


def score_agent(meta: dict, k: int, answer: str) -> dict:
    a = meta["agents"][k]
    seeds = [x["seed"] for x in meta["agents"]]
    rows, cols = meta["rows"], meta["cols"]
    n_true = len(meta["agents"])
    live = [l for l in a["links"] if l]
    torus = meta.get("topology") == "torus"
    true_role = "torus" if torus else role_of(len(live))

    count_txt = tag(answer, "count")
    m = re.search(r"\d+", count_txt or "")
    count = int(m.group(0)) if m else None
    s_count = max(0.0, 1 - abs(count - n_true) / n_true) if count is not None else 0.0

    arr_txt, pos_txt = tag(answer, "arrangement"), tag(answer, "position")
    s_arr = score_arrangement(arr_txt, rows, cols, torus)
    cells = parse_map(arr_txt) or parse_map(pos_txt)
    s_map, map_ok, map_bad = score_map(cells, meta)
    if torus:   # no cell is special: credit for seeing that there is no edge or corner to sit on
        c_role = "torus" if pos_txt and WRAP_WORDS.search(pos_txt.lower()) else claimed_role(pos_txt)
    else:
        c_role = claimed_role(pos_txt)
    s_pos = 1.0 if c_role == true_role else 0.0

    slots, s_wiring = [], []
    for kind, letter_other in (("p", "q"), ("q", "p")):
        for i in range(4):
            link = a["links"][i]
            state, nid, nport = parse_slot(tag(answer, f"{kind}{i}"), a["seed"], letter_other)
            if link is None:
                s = 1.0 if state == "dead" else 0.0
                truth = "dead"
            else:
                tid, tport = seeds[link["agent"]], link["port"]
                id_ok = state == "live" and nid == tid
                s = 0.5 * id_ok + 0.5 * (id_ok and nport == tport)
                truth = f"{tid} {letter_other}{tport}"
            claim = state if state != "live" else f"{nid} {letter_other}{'?' if nport is None else nport}"
            slots.append({"slot": f"{kind}{i}", "truth": truth, "claim": claim, "score": s})
            s_wiring.append(s)

    wiring = sum(s_wiring) / len(s_wiring)
    task = {}
    if meta["config"].get("task") == "sum":   # the network task: the exact sum of every private number
        mm = re.search(r"-?\d[\d,]*", tag(answer, "sum") or "")
        claim = int(mm.group(0).replace(",", "")) if mm else None
        task = {"sum_claim": claim, "sum_true": meta["answer"], "sum_score": float(claim == meta["answer"])}
        if meta.get("liars") is not None:            # the liar task: who and how many
            true_l = set(meta["liars"])
            mm = re.search(r"\d+", tag(answer, "liar_count") or "")
            lc = int(mm.group(0)) if mm else None
            said = set(re.findall(r"\b[0-9a-f]{8}\b", (tag(answer, "liars") or "").lower()))
            tp = len(said & true_l); prec = tp / len(said) if said else 0.0; rec = tp / len(true_l) if true_l else 1.0
            task.update(liar_count_claim=lc, liar_count_true=len(true_l), liar_count_score=float(lc == len(true_l)),
                        liars_claim=sorted(said), liars_true=sorted(true_l), liars_exact=float(said == true_l),
                        liars_precision=round(prec, 3), liars_recall=round(rec, 3), is_liar=bool(a.get("liar")))
    return {**task,
        "agent": k + 1, "seed": a["seed"], "pos": a["pos"], "true_role": true_role,
        "count": count, "count_score": round(s_count, 3),
        "arrangement": arr_txt, "arrangement_score": s_arr,
        "position": pos_txt, "claimed_role": c_role, "position_score": s_pos,
        "map_cells": {f"{r},{c}": v for (r, c), v in (cells or {}).items()},
        "map_score": round(s_map, 3), "map_correct": map_ok, "map_wrong": map_bad,
        "slots": slots, "wiring_score": round(wiring, 3),
        "total": round((s_count + s_arr + s_pos + s_map + wiring) / 5, 3),
    }


async def ask_all(meta: dict, ticks: list[dict], eval_tpl: str) -> tuple[int, list]:
    sim.set_width(meta["config"])
    states = sim.restore_states(meta, ticks)
    t = ticks[-1]["t"] if ticks else 0
    inputs = sim.read_inputs(meta["agents"], states)   # what they'd see if the run went on
    shown = meta["config"]["log_lines_shown"]
    turns = []
    import registers
    gluon_rows = {st.id for st, a in zip(states, meta["agents"]) if registers.muons_only(meta["config"]) and registers.is_gluon(a)}
    for st, inp in zip(states, inputs):
        if st.id in gluon_rows:
            continue                                           # a small process under muons_only: no model, no answer
        vals = {"t": t, "protocol": st.protocol.strip() or "(empty)",
                "recent_log": "\n".join(f"t{tt}: {l}" for tt, l in st.log[-shown:]) or "(empty)"}
        for i in range(4):
            vals[f"p{i}"], vals[f"q{i}"] = inp[i], st.outputs[i]
        images = []
        if sim.PIXEL:
            pv, images = sim.port_pictures(inp, st.outputs, int(meta["config"].get("pixel_scale", 16)))
            vals.update(pv)
        turns.append(sim.Turn(t=t + 1, state=st, inputs=inp, changed=[False] * 4,
                              history=list(st.convo) if meta["config"].get("memory") == "conversation" else [],
                              system=sim.system_prompt(meta, st.seed),
                              prompt=sim.fill(eval_tpl, vals), images=images))
    if os.environ.get("FOURFOLD_EVAL_BASE_URL"):      # fabe 2026-11-09: evaluate on another live server when the original is gone
        meta["config"]["llm"]["base_url"] = [os.environ["FOURFOLD_EVAL_BASE_URL"]]
        meta["config"]["llm"].pop("pools", None)
        print("eval: base_url overridden ->", meta["config"]["llm"]["base_url"], file=sys.stderr)
    agent = sim.make_agent(meta["config"])
    try:
        replies = await asyncio.gather(*(sim.safe_act(agent, tn) for tn in turns))
    finally:
        await agent.close()
    return t, list(zip(turns, replies))


def naming_report(meta: dict, ticks: list[dict]):
    """naming task, scored straight from the run (no model calls): how far the grid got
    toward one shared value, and which neighbours ended up agreeing."""
    agents = meta["agents"]
    best = sum(l is not None for a in agents for l in a["links"])
    totals = [sum(a.get("score", 0) for a in rec["agents"]) for rec in ticks]
    last = ticks[-1]
    q0 = [a["outputs"][0] for a in last["agents"]]
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(q0):
        if v.strip("0"):
            groups.setdefault(v, []).append(i + 1)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    # an edge is a pair of neighbours; agreed when both hold the same nonzero q0
    edges = {tuple(sorted((a["id"], l["agent"]))) for a in agents for l in a["links"] if l}
    agreed = sum(q0[x] == q0[y] and bool(q0[x].strip("0")) for x, y in edges)
    first_full = next((rec["t"] for rec, s in zip(ticks, totals) if s == best), None)
    peak = max(totals) if totals else 0
    print(f"\nnaming task after tick {last['t']}")
    print(f"  grid score now {totals[-1]}/{best}, peak {peak}"
          + (f", first full score at t{first_full}" if first_full else ", never full"))
    print(f"  distinct nonzero q0 values: {len(groups)}  (cluster sizes {sizes or '-'})")
    print(f"  neighbour pairs holding the same q0: {agreed}/{len(edges)}")
    for v, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    {v}  agents {ids}")
    row = lambda r: " ".join(f"{last['agents'][r * meta['cols'] + c]['score'] if 'score' in last['agents'][r * meta['cols'] + c] else '-'}"
                             f"/{sum(l is not None for l in agents[r * meta['cols'] + c]['links'])}"
                             for c in range(meta["cols"]))
    print("  per-agent score on the grid:")
    for r in range(meta["rows"]):
        print("    " + row(r))
    step = max(1, len(totals) // 50)
    marks = " .:-=+*#%@"
    spark = "".join(marks[min(len(marks) - 1, round(s / best * (len(marks) - 1)))] for s in totals[::step])
    print(f"  total over time: {spark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run id, run dir or ticks.jsonl (default: newest run)")
    ap.add_argument("--tick", type=int, help="evaluate as if the run ended after this tick")
    ap.add_argument("--show", action="store_true", help="print each agent's free-text answers")
    ap.add_argument("--naming", action="store_true",
                    help="naming task only: report agreement from the run without asking the agents anything")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the saved answers in eval-t<tick>.json without asking the agents again")
    args = ap.parse_args()
    sim.load_dotenv()

    path = resolve(args.run)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = next(l for l in lines if l.get("type") == "meta")
    ticks = [l for l in lines if l.get("type") == "tick"]
    if args.tick is not None:
        ticks = [r for r in ticks if r["t"] <= args.tick]
    if not ticks:
        sys.exit("no ticks to evaluate")
    eval_tpl = (ROOT / "prompts" / "eval.txt").read_text(encoding="utf-8")
    if meta["config"].get("task") == "sum" and int(meta["config"].get("liars") or 0):
        eval_tpl += ("<sum>the sum of the private numbers of the honest processes. one number.</sum>\n"
                     "<liar_count>how many processes are liars. one number.</liar_count>\n"
                     "<liars>the ids of the liars, separated by spaces. \"none\" if none.</liars>\n")
    elif meta["config"].get("task") == "sum":
        eval_tpl += "<sum>the sum of the private numbers of all processes. one number.</sum>\n"
    if args.naming:
        naming_report(meta, ticks)
        return

    t = ticks[-1]["t"]
    out = path.parent / f"eval-t{t}.json"
    scored = []
    if args.rescore:
        if not out.exists():
            sys.exit(f"--rescore: {out} not found")
        saved = json.loads(out.read_text(encoding="utf-8"))
        eval_tpl = saved.get("eval_prompt", eval_tpl)
        print(f"re-scoring saved answers for {meta['run_id']} after tick {t} ...")
        for old in saved["results"]:
            s = score_agent(meta, old["agent"] - 1, old["raw"])
            s["error"], s["raw"], s["prompt"] = old["error"], old["raw"], old["prompt"]
            scored.append(s)
    else:
        print(f"evaluating {meta['run_id']} after tick {t} with {meta['config']['agent']} ...")
        t, results = asyncio.run(ask_all(meta, ticks, eval_tpl))
        for tn, rep in results:
            s = score_agent(meta, tn.state.id, rep.raw)
            s["error"], s["raw"], s["prompt"] = rep.error, rep.raw, tn.prompt
            scored.append(s)

    out.write_text(json.dumps({"run_id": meta["run_id"], "tick": t, "time": time.time(),
                               "eval_prompt": eval_tpl, "results": scored}, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    n = len(meta["agents"])
    print(f"\n{'agent':>5} {'role':>7} {'claims':>7} {'count':>5} {'arr':>4} {'pos':>4} {'map':>9} {'wiring':>6} {'total':>6}")
    for s in scored:
        flag = "  ! " + s["error"][:60] if s["error"] else ""
        map_col = f"{s['map_score']:.2f} {s['map_correct']}/{s['map_wrong']}" if s["map_cells"] else "-"
        print(f"{s['agent']:>5} {s['true_role']:>7} {str(s['claimed_role']):>7} {str(s['count']):>5} "
              f"{s['arrangement_score']:>4.1f} {s['position_score']:>4.1f} {map_col:>9} "
              f"{s['wiring_score']:>6.2f} {s['total']:>6.2f}{flag}")
    mean = lambda key: sum(s[key] for s in scored) / len(scored)
    print(f"{'mean':>5} {'':>7} {'':>7} {'':>5} {mean('arrangement_score'):>4.2f} {mean('position_score'):>4.2f} "
          f"{mean('map_score'):>9.2f} {mean('wiring_score'):>6.2f} {mean('total'):>6.2f}   (true count {n}; map = score right/wrong)")

    if args.show:
        for s in scored:
            print(f"\n== agent {s['agent']} ({s['true_role']}, {s['seed']})")
            print(f"  count: {s['count']}\n  arrangement: {s['arrangement']}\n  position: {s['position']}")
            for sl in s["slots"]:
                mark = "ok" if sl["score"] == 1 else ("half" if sl["score"] else "--")
                print(f"  {sl['slot']}: claim {sl['claim']:<16} truth {sl['truth']:<14} {mark}")
    if meta["config"].get("task") == "naming":
        naming_report(meta, ticks)
    if meta["config"].get("task") == "sum":
        right = sum(s.get("sum_score", 0) for s in scored)
        claims = sorted((s.get("sum_claim") for s in scored), key=lambda x: (x is None, x))
        print(f"\ntask: sum of private numbers = {meta['answer']}; exactly right: {int(right)}/{len(scored)}")
        if meta.get("liars") is not None:
            honest = [x for x in scored if not x.get("is_liar")]
            print(f"  liars: {len(meta['liars'])} ({' '.join(meta['liars'])}); honest agents naming the exact set: "
                  f"{int(sum(x.get('liars_exact', 0) for x in honest))}/{len(honest)}; right count: {int(sum(x.get('liar_count_score', 0) for x in honest))}/{len(honest)}; "
                  f"honest agents with the honest sum: {int(sum(x.get('sum_score', 0) for x in honest))}/{len(honest)}")
        print("  answers given:", claims[:40], "..." if len(claims) > 40 else "")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
