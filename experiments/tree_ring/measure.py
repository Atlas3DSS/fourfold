"""what the brief asks of every arm, read off a run's ticks.jsonl and its eval file (no model calls).

  .venv/bin/python experiments/tree_ring/measure.py runs/<run> [runs/<run> ...]

an agent "holds" a total when a typed value says so: a "total" entry on one of its outputs (tree arms), its ring3 note
(ring arm). exact = that total is the true sum; wrong = it is not. the eval answers are counted the same way.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sim  # noqa: E402


def held(a, hint):
    """the totals an agent holds at this tick, as a set of sums."""
    out = set()
    if hint == "ring":
        v = (a.get("ring_res") or {}).get("3")
        if v:
            out.add(int(v[0]))
        else:
            v = (a.get("notes") or {}).get("ring3")
            if v and v.split()[0].isdigit():
                out.add(int(v.split()[0]))
    else:
        got = {}
        for v, k in zip(a["outputs"], a.get("out_types", [])):
            if k == "tree":
                for x, (kind, s, c) in sim.port_tree(v).items():
                    if kind == "total":
                        got.setdefault(x, set()).add(s)
        for x, sums in got.items():
            out |= sums
    return out


def measure(run):
    path = Path(run) / "ticks.jsonl" if Path(run).is_dir() else Path(run)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = next(l for l in lines if l.get("type") == "meta")
    ticks = [l for l in lines if l.get("type") == "tick"]
    cfg = meta["config"]; sim.set_width(cfg)
    n, ans, hint = len(meta["agents"]), meta["answer"], cfg.get("protocol_hint")
    two = hint == "tree" and len(cfg.get("tree_roots") or []) == 2
    first = allx = None
    ever_wrong, rejected, noreply, curve = set(), 0, 0, []
    for rec in ticks:
        exact = 0
        for a in rec["agents"]:
            h = held(a, hint)
            if two:      # both trees must be there and agree
                per = {}
                for v, k in zip(a["outputs"], a.get("out_types", [])):
                    if k == "tree":
                        for x, (kind, s, c) in sim.port_tree(v).items():
                            if kind == "total":
                                per[x] = s
                ok = len(per) == 2 and per["A"] == per["B"] == ans
            else:
                ok = h == {ans}
            exact += ok
            if h - {ans}:
                ever_wrong.add(a["agent"])
            rejected += sum("rejected" in x for x in a.get("actions", []))
            noreply += bool(a["errors"]) and not a.get("raw")
        curve.append(exact)
        if exact and first is None:
            first = rec["t"]
        if exact == n and allx is None:
            allx = rec["t"]
    def toks(t):
        r = next((x for x in ticks if x["t"] == t), None)
        v = [a["usage"].get("prompt_tokens") or a["usage"].get("input_tokens") for a in (r["agents"] if r else []) if a.get("usage")]
        v = [x for x in v if x]
        return int(statistics.median(v)) if v else None
    def chars(t):
        r = next((x for x in ticks if x["t"] == t), None)
        v = [len(a.get("prompt") or "") for a in (r["agents"] if r else [])]
        return int(statistics.median(v)) if v else None
    secs = [b["time"] - a["time"] for a, b in zip(ticks, ticks[1:])]
    out = {"run": meta["run_id"], "agents": n, "ticks": len(ticks), "true_sum": ans, "first_exact": first, "all_exact": allx,
           "exact_at_end": curve[-1] if curve else 0, "agents_ever_holding_a_wrong_total": len(ever_wrong),
           "rejected_writes": rejected, "no_reply_turns": noreply,
           "median_prompt_tokens": {f"t{t}": toks(t) for t in (1, 10, 30, 60)},
           "median_tick_prompt_chars": {f"t{t}": chars(t) for t in (1, 10, 30, 60)},
           "median_seconds_per_tick": round(statistics.median(secs), 1) if secs else None, "exact_by_tick": curve}
    evs = sorted(path.parent.glob("eval-t*.json"), key=lambda p: int(p.stem.split("-t")[1]))
    for e in evs:
        res = json.loads(e.read_text(encoding="utf-8"))["results"]
        claims = [r.get("sum_claim") for r in res]
        counts = sorted(r.get("count") for r in res if r.get("count") is not None)
        out[f"eval_{e.stem.split('-')[1]}"] = {"exact": sum(c == ans for c in claims), "wrong": sum(c is not None and c != ans for c in claims),
                                              "no_number": sum(c is None for c in claims), "wrong_values": sorted({c for c in claims if c is not None and c != ans})[:20],
                                              "count_answers_min_median_max": [counts[0], counts[len(counts) // 2], counts[-1]] if counts else None,
                                              "count_exact": sum(c == n for c in counts)}
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(measure(run), indent=1))
