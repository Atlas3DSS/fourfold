"""which agents wrote readable text on their ports, per run.

  python experiments/quincy/ascii_text.py RUN_DIR [RUN_DIR ...] [--window 11] [--show 12]

A port value counts as text when its bytes, with zero padding stripped, are all printable ASCII, at least 4 long,
with at least 3 letters and at least 3 different characters (so fills like 3333 or UUUU don't count). Only values
the simulator accepted are seen here: a value of the wrong length was rejected and never reached a neighbour.

Per run: agents that wrote text in the first --window ticks (the same window for every run, so runs of
different lengths compare fairly), agents over the whole run, how many question texts were arriving on inputs
at the last tick, and the first --show distinct texts with the tick and agent that first sent each.
Agents are numbered from 1, as in eval.py, view.py and erz0r's README (agent k is index k-1 in ticks.jsonl).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

QUESTION = re.compile(rb"(?i)\b(what|who|where) (are|am)\b")


def text(v: str) -> str | None:
    b = bytes.fromhex(v).strip(b"\x00")
    if len(b) < 4 or not all(32 <= c < 127 for c in b):
        return None
    s = b.decode()
    if len(re.findall(r"[A-Za-z]", s)) < 3 or len(set(s.replace(" ", ""))) < 3:
        return None
    return s


def load(run: Path):
    f = run / "ticks.jsonl"
    fh = open(f, encoding="utf-8") if f.exists() else gzip.open(run / "ticks.jsonl.gz", "rt", encoding="utf-8")
    with fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    return lines[0], [l for l in lines if l.get("type") == "tick"]


def scan(ticks):
    agents, first = set(), {}
    for t in ticks:
        for a in t["agents"]:
            for v in a["outputs"]:
                s = text(v)
                if s:
                    agents.add(a["agent"])
                    first.setdefault(s, (t["t"], a["agent"] + 1))   # 1-based, like eval.py, view.py and erz0r's README
    return agents, first


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    window = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 11
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 12
    args = [a for a in args if not a.isdigit()]
    for run in map(Path, args):
        meta, ticks = load(run)
        a_win, _ = scan(ticks[:window])
        a_all, first = scan(ticks)
        heard = sum(1 for a in ticks[-1]["agents"] for v in a["inputs"] if QUESTION.search(bytes.fromhex(v)))
        print(f"{run.name}: {len(meta['agents'])} agents, {ticks[-1]['t']} ticks, prime {meta['config'].get('prime')!r}")
        print(f"  wrote text in the first {min(window, len(ticks))} ticks: {len(a_win)} agents;"
              f" whole run: {len(a_all)}; question texts arriving on inputs at t{ticks[-1]['t']}: {heard}")
        for s, (t, ag) in sorted(first.items(), key=lambda kv: kv[1])[:show]:
            print(f"    t{t} agent {ag}: {s!r}")


if __name__ == "__main__":
    main()
