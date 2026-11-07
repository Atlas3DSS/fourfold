"""summary numbers for the Quincy runs, straight from each run's ticks.jsonl(.gz) and eval json.

  python experiments/quincy/analyze.py experiments/quincy/runs/*          # table
  python experiments/quincy/analyze.py RUN_DIR --json                     # one run, as json

per run:
  tick_s        median wall time per tick
  err_rate      share of agent-turns with any rejected or missing value
  active        share of agents whose outputs changed, mean over the last 10 ticks
  social        share of agents whose protocol or log talks about other processes, at the last tick
  hears_nb      share of agents with a true neighbour's 8-hex id arriving on an input, last tick
  names_nb      share of agents that write a true neighbour's id into their protocol or log, last tick
  eval_*        mean eval scores (count, arrangement, position, map, wiring, total) if the run was scored
  says_alone    share of agents whose eval count answer is 1
"""
from __future__ import annotations

import gzip
import json
import re
import statistics as st
import sys
from pathlib import Path

SOCIAL = re.compile(r"(another|other|neighbo\w*|peer|external|remote) (process|agent|entity|node|sender)"
                    r"|not alone|someone|communicat\w*|handshake|\bmessage")


def load(run: Path):
    f = run / "ticks.jsonl"
    opener = open if f.exists() else gzip.open
    f = f if f.exists() else run / "ticks.jsonl.gz"
    with opener(f, "rt", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    meta = next(l for l in lines if l["type"] == "meta")
    ticks = [l for l in lines if l["type"] == "tick"]
    evals = sorted(run.glob("eval-t*.json"), key=lambda p: int(p.stem.split("-t")[1]))
    return meta, ticks, (json.loads(evals[-1].read_text()) if evals else None)


def summarize(run: Path) -> dict:
    meta, ticks, ev = load(run)
    cfg = meta["config"]
    seeds = [a["seed"] for a in meta["agents"]]
    nb = [{l["agent"] for l in a["links"] if l} for a in meta["agents"]]
    n = len(seeds)
    times = [b["time"] - a["time"] for a, b in zip(ticks, ticks[1:])]
    turns = [a for t in ticks for a in t["agents"]]
    act = [sum(x["outputs"] != y["outputs"] for x, y in zip(p["agents"], t["agents"])) / n
           for p, t in zip(ticks, ticks[1:])][-10:]
    last = ticks[-1]["agents"]
    notes = [(a["protocol"] + " " + " ".join(a["log"])).lower() for a in last]
    out = {
        "run": run.name, "grid": f"{meta['rows']}x{meta['cols']}", "topology": meta.get("topology", "grid"),
        "agents": n, "ticks": ticks[-1]["t"],
        "thinking": "off" if (cfg.get("llm", {}).get("extra_body", {}).get("chat_template_kwargs", {})
                              .get("enable_thinking") is False) else
                    cfg.get("llm", {}).get("extra_body", {}).get("reasoning_effort", "on"),
        "prime": cfg.get("prime", ""),
        "tick_s": round(st.median(times), 1) if times else None,
        "err_rate": round(sum(bool(a["errors"]) for a in turns) / len(turns), 3),
        "active": round(st.mean(act), 2) if act else None,
        "social": round(sum(bool(SOCIAL.search(x)) for x in notes) / n, 2),
        "hears_nb": round(sum(any(seeds[j] in v for v in last[i]["inputs"] for j in nb[i]) for i in range(n)) / n, 2),
        "names_nb": round(sum(any(seeds[j] in notes[i] for j in nb[i]) for i in range(n)) / n, 2),
    }
    if ev:
        R = ev["results"]
        for k in ("count_score", "arrangement_score", "position_score", "map_score", "wiring_score", "total"):
            out["eval_" + k.replace("_score", "")] = round(st.mean(r[k] for r in R), 3)
        out["says_alone"] = round(sum(r["count"] == 1 for r in R) / len(R), 2)
        out["eval_tick"] = ev["tick"]
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = [summarize(Path(a)) for a in args]
    if "--json" in sys.argv:
        print(json.dumps(rows if len(rows) > 1 else rows[0], indent=1))
        return
    cols = ["run", "grid", "topology", "agents", "ticks", "thinking", "tick_s", "err_rate", "active", "social",
            "hears_nb", "names_nb", "says_alone", "eval_count", "eval_wiring", "eval_total"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    main()
