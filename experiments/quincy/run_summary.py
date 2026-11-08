"""plain-language summary of one run: what it is, what the agents were told, what they were asked,
and how it ended (did they work out the answers?). stdlib only; used by viz_data.py for the viewer's
"about this run" panel and by fourfold-board/collect.py for the run list.

  python experiments/quincy/run_summary.py RUN_DIR [RUN_DIR ...] [--notes labels.json] [--json]

Everything is computed from ticks.jsonl(.gz), eval-t*.json and ratchet.jsonl; nothing is judged by a model.
"final state" numbers come from the scorer when the run was scored (eval.py asks every agent the questions and
scores them by code), otherwise from the agents' own notes at the last tick, and the text says which.

--notes FILE: optional {"<run>": {"title": "...", "note": "..."}} hand-written additions per run.
"""
from __future__ import annotations

import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path

TEXT_LETTERS = re.compile(r"[A-Za-z]")
QUESTION = re.compile(r"(?i)\b(what|who|where) (are|am)\b")
FIVE = ["what are you?", "who am i?", "who are we?", "what are we?", "where are we?"]
OTHERS = re.compile(r"not alone|other process|another process|other agent|other node|neighbou?r|peer|"
                    r"external (?:process|source|agent|sender)")
ALONE = re.compile(r"\balone\b|isolated|single process|sole process|self[- ]?feedback|closed loop|only process")
CLAIM = re.compile(r"(?i)(?:final answer|final ans|answer|sum)\s*[:=]?\s*(?:is\s*)?(\d[\d,]{2,})")
NPROC = re.compile(r"(?i)\b(\d{1,3})\s*(?:other\s+)?(?:processes|procs|agents|nodes|peers)\b")
LIVE_SECONDS = 900


def text(v: str) -> str | None:
    """the ascii_text.py rule: printable, >= 4 chars, >= 3 letters, >= 3 distinct characters."""
    b = bytes.fromhex(v).strip(b"\x00")
    if len(b) < 4 or not all(32 <= c < 127 for c in b):
        return None
    s = b.decode()
    if len(TEXT_LETTERS.findall(s)) < 3 or len(set(s.replace(" ", ""))) < 3:
        return None
    return s


def load(run: Path):
    f = run / "ticks.jsonl"
    fh = open(f, encoding="utf-8") if f.exists() else gzip.open(run / "ticks.jsonl.gz", "rt", encoding="utf-8")
    with fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    meta = next(l for l in lines if l["type"] == "meta")
    ticks = [l for l in lines if l["type"] == "tick"]
    evals = sorted(run.glob("eval-t*.json"), key=lambda p: int(p.stem.split("-t")[1]))
    ev = json.loads(evals[-1].read_text(encoding="utf-8")) if evals else None
    rf = run / "ratchet.jsonl"
    ratchet = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()] if rf.exists() else []
    return meta, ticks, ev, ratchet


def n_fmt(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def plural(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


# ---------------------------------------------------------------- what the run is

MODEL_NAMES = {"qwen38-27b": "Qwen3.8-27B", "claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
               "claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1"}


def model_name(cfg: dict) -> str:
    """the agents' model in plain words, from the agent kind's own config block."""
    kind = cfg.get("agent", "llm")
    if kind == "echo":
        return "echo agents (no model)"
    raw = (cfg.get(kind) or {}).get("model") or ""
    name = MODEL_NAMES.get(raw, raw or "unknown model")
    if kind == "claudecode":
        return f"{name} via Claude Code"
    if kind == "anthropic":
        return f"{name} via the API"
    return name




def told(cfg: dict) -> tuple[str, str]:
    """-> (short tag for the title, full sentence for the panel)."""
    prime = (cfg.get("prime") or "").strip()
    parts, sentence = [], []
    if cfg.get("task") == "sum":
        parts.append("Sum task")
        sentence.append("Each agent holds a private number and must report the sum of everyone's numbers.")
        if cfg.get("actions"):
            parts.append("5 actions")
            sentence.append("It can say, ask, remember, recall and compute.")
        if cfg.get("modules"):
            parts.append("arrays with slow seams" if (cfg.get("seam_period") or 1) > 1 else "arrays")
        if cfg.get("memory") == "conversation":
            parts.append("long memory")
            sentence.append("Each agent is one growing conversation that remembers every tick.")
    if prime:
        if "ascii" in prime.lower():
            parts.append("Hint + how to send ASCII")
        elif "five questions" in prime.lower():
            parts.append("Hint + five questions")
        else:
            parts.append("Hint: “you might not be alone”")
        sentence.append(f"Told: “{prime}”")
    if not parts:
        parts.append("No hint (original prompt)")
        sentence.append("Nothing beyond erz0r's original prompt: no hint that anyone else exists, no way to talk.")
    think = (cfg.get("llm") or {}).get("extra_body", {})
    if "reasoning_effort" in think:
        parts.append(f"thinking {think['reasoning_effort']}")
        sentence.append(f"Reasoning on ({think['reasoning_effort']} effort).")
    elif think.get("chat_template_kwargs", {}).get("enable_thinking") is False or \
            (cfg.get(cfg.get("agent", "llm")) or {}).get("thinking") == "disabled":
        sentence.append("Thinking off.")
    return ", ".join(parts), " ".join(sentence)


def world(meta: dict) -> str:
    cfg = meta["config"]
    n = len(meta["agents"])
    shape = "torus (edges wrap, nobody sits at an edge)" if meta.get("topology") == "torus" else "flat grid (edge ports are dead)"
    s = f"{n} agents on a {meta['rows']}×{meta['cols']} {shape}"
    if cfg.get("modules"):
        k = (meta["rows"] // cfg["modules"][0]) * (meta["cols"] // cfg["modules"][1])
        s += f", cut into {k} arrays of {cfg['modules'][0]}×{cfg['modules'][1]}"
        if (cfg.get("seam_period") or 1) > 1:
            s += f"; a link between two arrays carries a new value only every {cfg['seam_period']} ticks"
    return s + "."


def asked(meta: dict) -> str:
    n = len(meta["agents"])
    s = (f"At the end every agent is asked, and scored by code: how many processes like you exist (truth {n})? "
         f"How are they arranged (a {meta['rows']}×{meta['cols']} {meta.get('topology', 'grid')})? Where do you sit? "
         f"Which process feeds each of your ports?")
    if meta["config"].get("task") == "sum":
        s += f" And the task: the exact sum of all private numbers ({n_fmt(meta['answer'])})."
    return s


def status(meta: dict, ticks: list, ev, ratchet: list, run: Path) -> dict:
    f = run / "ticks.jsonl"
    mtime = f.stat().st_mtime if f.exists() else (run / "ticks.jsonl.gz").stat().st_mtime
    t = ticks[-1]["t"]
    planned = meta["config"].get("ticks")
    live = time.time() - mtime < LIVE_SECONDS
    evicted = next((r for r in ratchet if not r.get("keep") and r["tick"] == t), None)
    if live and not evicted:
        word, line = "running", f"Running, tick {t}" + (f" of {planned}" if planned else "")
    elif evicted:
        word, line = "evicted", f"Stopped by the ratchet at tick {evicted['tick']}: nothing new in the last {evicted['window']} ticks"
    elif ev and ev["tick"] == t:
        word, line = "done", f"Finished at tick {t}, scored"
    elif planned and t >= planned:
        word, line = "done", f"Finished at tick {t}, not scored yet"
    else:
        word, line = "paused", f"Paused at tick {t}" + (f" of {planned}" if planned else "")
    if ev and ev["tick"] != t:
        line += f" (scored at tick {ev['tick']})"
    if evicted and ev and ev["tick"] == t:
        line += ", scored"
    return {"state": word, "line": line, "ticks": t, "planned": planned, "scored_at": ev["tick"] if ev else None,
            "last_tick_time": ticks[-1]["time"], "evicted": evicted}


# ---------------------------------------------------------------- how it ended


def talk(meta: dict, ticks: list) -> dict:
    n = len(meta["agents"])
    writers, first = set(), {}
    for rec in ticks:
        for a in rec["agents"]:
            for v in a["outputs"]:
                s = text(v)
                if s:
                    writers.add(a["agent"])
                    first.setdefault(s, (rec["t"], a["agent"] + 1))
    last = ticks[-1]["agents"]
    on_wire = collections.Counter(s for a in last for v in a["outputs"] if (s := text(v)))
    hearing = sum(1 for a in last if any(text(v) for v in a["inputs"]))
    msgs = sorted(first.items(), key=lambda kv: kv[1])
    return {"writers": len(writers), "n": n, "distinct": len(first), "hearing": hearing,
            "on_wire": on_wire.most_common(8), "first": [[t, ag, s] for s, (t, ag) in msgs[:14]],
            "questions_sent": [q for q in FIVE if any(q.lower() in s.lower() for s in first)],
            "statements": [[t, ag, s] for s, (t, ag) in msgs if not QUESTION.search(s)]}


def beliefs(meta: dict, ticks: list) -> dict:
    """what the agents' own notes say at the last tick (no model call): others / alone / undecided."""
    last = ticks[-1]["agents"]
    c = collections.Counter()
    for a in last:
        p = a["protocol"].lower()
        c["others" if OTHERS.search(p) else "alone" if ALONE.search(p) else "undecided"] += 1
    guesses = collections.Counter()
    for a in last:
        m = NPROC.findall(a["protocol"])
        if m:
            guesses[int(m[-1])] += 1
    return {"others": c["others"], "alone": c["alone"], "undecided": c["undecided"], "guesses": guesses.most_common(6)}


def sum_state(meta: dict, ticks: list) -> dict:
    agents = meta["agents"]
    total = meta["answer"]
    mods = collections.defaultdict(int)
    for a in agents:
        mods[a.get("module", 0)] += a["secret"]
    last = ticks[-1]["agents"]
    claims = []
    for a in last:
        m = CLAIM.findall(a["protocol"])
        claims.append(int(m[-1].replace(",", "")) if m else None)
    exact = sum(1 for a in last if str(total) in (a["protocol"] + " " + (a.get("raw") or "")))
    cnt = collections.Counter(c for c in claims if c)
    secrets = {a["secret"] for a in agents}
    known = sorted(len({int(x) for x in re.findall(r"\b(\d{1,3})\b", a["protocol"]) if int(x) in secrets}) for a in last)
    array_hits = sum(1 for c in claims if c in mods.values())
    near = None
    if cnt and len(mods) > 1:
        v, _ = min(((v, min(abs(v - s) for s in mods.values())) for v in cnt), key=lambda x: x[1])
        s = min(mods.values(), key=lambda s: abs(s - v))
        near = {"claim": v, "array_sum": s}
    first_exact = next((rec["t"] for rec in ticks if any(str(total) in a["protocol"] for a in rec["agents"])), None)
    return {"total": total, "exact_in_notes": exact, "first_exact_tick": first_exact, "no_claim": claims.count(None),
            "distinct": len(cnt), "top": cnt.most_common(5), "array_sum_hits": array_hits, "nearest": near,
            "known_median": known[len(known) // 2] if known else 0, "known_max": known[-1] if known else 0,
            "array_sums": dict(mods)}


def eval_state(meta: dict, ev: dict) -> dict:
    R = ev["results"]
    n = len(meta["agents"])
    counts = collections.Counter(r["count"] for r in R)
    out = {"tick": ev["tick"], "n": len(R), "alone": counts.get(1, 0), "right_count": counts.get(n, 0),
           "counts": sorted(((str(k) if k is not None else "no number", v) for k, v in counts.items()), key=lambda kv: -kv[1]),
           "maps": sum(1 for r in R if r["map_cells"]), "arrangement_right": sum(1 for r in R if r["arrangement_score"] >= 1),
           "wraps": sum(1 for r in R if r["arrangement_score"] >= 0.5) if meta.get("topology") == "torus" else None,
           "mean_total": round(sum(r["total"] for r in R) / len(R), 3),
           "mean_wiring": round(sum(r["wiring_score"] for r in R) / len(R), 2)}
    if "sum_claim" in R[0]:
        sc = collections.Counter(r["sum_claim"] for r in R)
        out["sum_right"] = sum(1 for r in R if r.get("sum_score"))
        out["sum_claims"] = sorted(((n_fmt(k) if k is not None else "no number", v) for k, v in sc.items()), key=lambda kv: -kv[1])
    return out


def summarize(run: Path, notes: dict | None = None) -> dict:
    meta, ticks, ev, ratchet = load(run)
    return summarize_loaded(run, meta, ticks, ev, ratchet, notes)


def summarize_loaded(run: Path, meta: dict, ticks: list, ev, ratchet: list, notes: dict | None = None) -> dict:
    cfg = meta["config"]
    n = len(meta["agents"])
    notes = notes or {}
    tag, told_s = told(cfg)
    st = status(meta, ticks, ev, ratchet, run)
    if notes.get("status"):
        st["line"] = notes["status"]
    shape = "torus" if meta.get("topology") == "torus" else "grid"
    title = notes.get("title") or f"{tag} · {n} agents, {meta['rows']}×{meta['cols']} {shape} · {model_name(cfg)}"
    tk = talk(meta, ticks)
    be = beliefs(meta, ticks)
    findings: list[str] = []
    verdict = ""
    is_sum = cfg.get("task") == "sum"

    # ---- the scored answers, when the run was scored
    if ev:
        es = eval_state(meta, ev)
        src = f"When asked at tick {es['tick']}"
        if is_sum:
            verdict = (f"No. The true sum is {n_fmt(meta['answer'])}; {es['sum_right']} of {es['n']} agents gave it."
                       if es["sum_right"] < es["n"] else f"Yes: all {es['n']} agents gave the exact sum {n_fmt(meta['answer'])}.")
            findings.append(f"{src}, the sums they gave: " + ", ".join(f"{k} ({plural(v, 'agent')})" for k, v in es["sum_claims"][:6]) + ".")
        else:
            if es["alone"] == es["n"]:
                verdict = f"No. All {es['n']} agents concluded they were alone (answered 1; the truth is {n})."
            elif es["right_count"]:
                verdict = f"Partly: {es['right_count']} of {es['n']} agents gave the right count ({n})."
            else:
                verdict = f"No. Nobody gave the right count ({n}); {es['alone']} of {es['n']} said they were alone."
        findings.append(f"{src}, “how many processes are there?” (truth {n}) got: "
                        + ", ".join(f"{k} ({plural(v, 'agent')})" for k, v in es["counts"][:6]) + ".")
        arr = f"{es['arrangement_right']} named the right arrangement"
        if es["wraps"] is not None:
            arr += f"; {es['wraps']} earned any arrangement credit (right size or said the edges wrap)"
        findings.append(f"{arr}; {es['maps']} drew a map. Mean score {es['mean_total']} (erz0r's sol showcase: 0.60).")
        if es["mean_wiring"] >= 0.15 and not is_sum:
            findings.append(f"Wiring score {es['mean_wiring']}: mostly credit for calling dead edge ports dead.")
    notes_src = f"In their notes at tick {st['ticks']}" + ("" if ev else " (not scored yet)")

    # ---- the task, from the notes
    if is_sum:
        ss = sum_state(meta, ticks)
        if not ev:
            verdict = f"Not so far. The true sum is {n_fmt(ss['total'])}; {ss['exact_in_notes']} of {n} agents have it in their notes at tick {st['ticks']}."
            if ss["top"] and ss["exact_in_notes"] < n:
                verdict += f" {plural(ss['distinct'], 'different sum')} are in circulation; the most common, {n_fmt(ss['top'][0][0])}, is held by {plural(ss['top'][0][1], 'agent')}."
            if ss["exact_in_notes"] == n:
                verdict = f"Yes: all {n} agents have {n_fmt(ss['total'])} in their notes."
        line = f"{notes_src}: {plural(ss['distinct'], 'different sum')} in circulation"
        if ss["top"]:
            line += ", most common " + ", ".join(f"{n_fmt(v)} ({k})" for v, k in ss["top"][:3])
        line += f"; {ss['no_claim']} agents have no sum yet."
        findings.append(line)
        if ss["array_sums"] and len(ss["array_sums"]) > 1:
            line = f"The seven arrays' own sums are {', '.join(n_fmt(v) for v in ss['array_sums'].values())}; {ss['array_sum_hits']} agents hold exactly one of them"
            if ss["nearest"]:
                line += f" (closest claim {n_fmt(ss['nearest']['claim'])} against an array's {n_fmt(ss['nearest']['array_sum'])})"
            findings.append(line + ".")
        findings.append(f"Each agent's notes hold a median of {ss['known_median']} of the {n} private numbers (most: {ss['known_max']}).")
        if be["guesses"]:
            findings.append("How many processes they think exist (truth " + str(n) + "): "
                            + ", ".join(f"{k} ({plural(v, 'agent')})" for k, v in be["guesses"][:4]) + ".")

    # ---- talk
    if tk["writers"]:
        line = f"{tk['writers']} of {n} agents wrote readable text on a port ({plural(tk['distinct'], 'distinct message')})"
        line += f"; at the last tick {tk['hearing']} agents had text arriving on an input."
        findings.append(line)
    else:
        findings.append(f"No agent wrote readable text on a port in {st['ticks']} ticks.")
    if "five questions" in (cfg.get("prime") or "").lower():
        qs = tk["questions_sent"]
        findings.append(f"Questions that went out on the wire: {len(qs)} of the five" + (f" ({', '.join(qs)})" if qs else "") + ".")
        answers = []
        for _, _, s in tk["statements"]:
            s = s.strip().rstrip(".")
            if len(s.split()) >= 2 and re.search(r"(?i)\b(we are|i am a|in a|here|process|simulat)", s) and not re.search(r"[0-9a-f]{8}", s):
                if s.lower() not in [a.lower() for a in answers] and not any(a.lower().startswith(s.lower()) for a in answers):
                    answers = [a for a in answers if not s.lower().startswith(a.lower())] + [s]
        if answers:
            findings.append("Answers that went out: " + ", ".join(f"“{s}”" for s in answers[:7]) + ".")
        if not ev:
            verdict = (f"Not yet asked. So far they ask each other the questions and answer with " + ", ".join(f"“{s}”" for s in answers[:4]) +
                       f"; nobody has worked out that there are {n} of them or that the world wraps." if answers else
                       f"Not yet asked. They ask the questions, but no answers have gone out yet.")
    elif not is_sum and not ev:
        verdict = (f"Not yet asked. At tick {st['ticks']}, {be['others']} of {n} agents' notes mention other processes, "
                   f"{be['alone']} say they are alone, {be['undecided']} are undecided; " +
                   ("nobody has written text on a port." if not tk["writers"] else f"{tk['writers']} have written text."))
    if not is_sum and not ev:
        findings.append(f"Notes at tick {st['ticks']}: {be['others']} mention other processes, {be['alone']} say alone or closed loop, "
                        f"{be['undecided']} undecided.")
    if st["evicted"]:
        findings.append(f"The ratchet stopped the run at tick {st['evicted']['tick']}: no new talk, discovery or task progress in {st['evicted']['window']} ticks.")
    if notes.get("note"):
        findings.append(notes["note"])

    asked_s = asked(meta)
    reg = cfg.get("registers") or {}
    if reg.get("enabled"):               # a different kind of run: the questions above don't apply to it
        rows = [r["reg"] for r in ticks if r.get("reg")]
        events = [e for r in rows for e in r.get("events", [])]
        count = lambda kind: sum(e["kind"] == kind for e in events)
        delivered = rows[-1]["delivered"] if rows else 0
        shown = (ticks[-1]["t"] - 1) // max(1, int(reg.get("input_period", 12))) + 1 if ticks else 0
        help_bits = [x for x, on in (("told where the input and output are", reg.get("tell_cells")),
                                     ("shown the wiring", reg.get("show_wiring"))) if on]
        told_s = ("They have addresses and small registers, and they are told the job. " +
                  (f"Seam nodes are {' and '.join(help_bits)}." if help_bits else ""))
        asked_s = "The job: get each value from an input at one spot to an output at another, exactly, while it is still current."
        verdict = (f"Yes for {delivered} of {shown} values." if delivered else f"Not yet: 0 of {shown} values arrived.")
        findings = [f"{plural(count('program_trial'), 'instruction')} tried: {count('program_kept')} kept, "
                    f"{count('program_reverted') + count('program_failed_test')} undone or failed."]
        if notes.get("note"):
            findings.append(notes["note"])
    return {
        "run": run.name, "title": title, "world": world(meta), "told": told_s, "asked": asked_s,
        "status": st["line"], "state": st["state"], "ticks": st["ticks"], "scored_at": st["scored_at"],
        "verdict": verdict, "findings": findings, "messages": tk["first"],
        "label": f"{title} · {st['line'].split(' (')[0].lower()}",
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    notes = {}
    if "--notes" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--notes") + 1])
        notes = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        args = [a for a in args if a != str(f)]
    out = [summarize(Path(a), notes.get(Path(a).name)) for a in args]
    if "--json" in sys.argv:
        print(json.dumps(out if len(out) > 1 else out[0], indent=1, ensure_ascii=False))
        return
    for s in out:
        print(f"\n## {s['title']}   [{s['run']}]\n{s['status']}\n{s['world']} {s['told']}\n{s['asked']}\n\nDid they figure it out? {s['verdict']}")
        for f in s["findings"]:
            print(f"- {f}")
        if s["messages"]:
            print("  first messages: " + "; ".join(f"t{t} #{a} {m!r}" for t, a, m in s["messages"][:8]))


if __name__ == "__main__":
    main()
