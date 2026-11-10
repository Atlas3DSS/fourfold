"""perfect, no-llm agents for the sum task under the bridge: the tick floor of each method, through the real plumbing.

  flood  every agent passes on id:number pairs, two a port a tick, the oldest it has not yet sent on that port.
  tree   convergecast: the rules of sim.TREE_TEXT, read off the prompt exactly as an llm would see them.
  ring   one ring per port (the tesseract): the rules of sim.RING_TEXT.

an agent sees only its system prompt and tick prompt (it parses them), and writes only with the typed actions. tree and
ring keep no memory of their own: everything they need is in the prompt (inputs, own outputs, notes), which is the
point of those two methods. flood keeps what an llm would keep in its notes: the pairs it knows and how far each port
has got. miss_rate: the chance an agent gives no reply on a tick (its ports keep their values).
"""
from __future__ import annotations

import random
import re

import sim

_IN = re.compile(r"^p([0-3]) [0-9a-f]+  \((.*)\)$", re.M)
_OUT = re.compile(r"^q([0-3]): (.*)$", re.M)
_NOTE = re.compile(r"^(\w+) = (.*)$", re.M)


def entries(decoded: str, kind: str):
    """'tree A wave; B sub 12 3' -> {'A': ('wave',), 'B': ('sub', 12, 3)}; 'ring id s c; ...' -> [(id, s, c)];
    'pairs id:n, id:n' -> [(id, n)]. anything else -> empty."""
    if kind == "tree":
        out = {}
        if decoded.startswith("tree "):
            for it in decoded[5:].split(";"):
                w = it.split()
                if len(w) >= 2 and w[1] in sim.TREE_KINDS:
                    out[w[0]] = (w[1], *map(int, w[2:4]))
        return out
    if kind == "ring":
        return [(w[0], int(w[1]), int(w[2])) for w in (it.split() for it in decoded[5:].split(";"))
                if len(w) == 3] if decoded.startswith("ring ") else []
    return [(a, int(b)) for a, b in re.findall(r"([0-9a-f]{8}):(\d+)", decoded)] if decoded.startswith("pairs ") else []


class ScriptedAgent:
    def __init__(self, cfg: dict):
        self.kind = cfg.get("kind", "flood")
        self.miss = float(cfg.get("miss_rate") or 0.0)
        self.mem: dict = {}                       # flood only: seed -> {"order": [(id, n)], "sent": [0, 0, 0, 0]}

    async def act(self, turn: sim.Turn) -> sim.Reply:
        if self.miss and random.Random(f"{turn.state.seed}:miss:{turn.t}").random() < self.miss:
            return sim.Reply(error="scripted miss: no reply this tick")
        me = re.search(r"assigned to you: ([0-9a-f]{8})", turn.system).group(1)
        number = int(re.search(r"yours is (\d+)", turn.system).group(1))
        status = {int(i): s.split("; ", 1)[1] if "; " in s else "" for i, s in _IN.findall(turn.prompt)}
        ins = [status.get(i, "") for i in range(4)]
        body = turn.prompt.split("your output ports hold now:", 1)
        outs = dict((int(i), d) for i, d in _OUT.findall(body[1].split("\narrived on", 1)[0])) if len(body) == 2 else {}
        notes = dict(_NOTE.findall(turn.prompt.split("your notes (key = value):", 1)[1].split("\nresults of", 1)[0])) \
            if "your notes (key = value):" in turn.prompt else {}
        acts, answer = getattr(self, self.kind)(turn, me, number, ins, [outs.get(i, "empty") for i in range(4)], notes)
        raw = f"<log>scripted {self.kind}</log>\n"
        if answer:
            raw += f"<protocol>{answer}</protocol>\n"
        return sim.Reply(raw=raw + "\n".join(acts))

    async def close(self):
        pass

    # ---------------------------------------------------------------- flood

    def flood(self, turn, me, number, ins, outs, notes):
        m = self.mem.setdefault(me, {"order": [(me, number)], "sent": [0] * 4})
        known = {i for i, _ in m["order"]}
        for d in ins:
            for i, n in entries(d, "pairs"):
                if i not in known:
                    known.add(i); m["order"].append((i, n))
        acts = []
        for p in range(4):
            batch = m["order"][m["sent"][p]:m["sent"][p] + sim.HEXN // 16]
            if batch:
                m["sent"][p] += len(batch)
                acts.append(f'<write port="q{p}" as="pairs">' + ", ".join(f"{i}:{n}" for i, n in batch) + "</write>")
        return acts, f"sum so far {sum(n for _, n in m['order'])} over {len(m['order'])} processes"

    # ---------------------------------------------------------------- tree

    def tree(self, turn, me, number, ins, outs, notes):
        tin = [entries(d, "tree") for d in ins]
        tout = [entries(d, "tree") for d in outs]
        acts, totals = {}, {}
        letters = sorted(set(re.findall(r"tree ([AB]) starts", turn.system) + re.findall(r"root of tree ([AB])", turn.system))) \
            or ["A"]
        if "both trees start" in turn.system:
            letters = ["A", "B"]
        for x in letters:
            root = f"you are the root of tree {x}" in turn.system
            total = next((e[x] for e in tin if e.get(x, ("",))[0] == "total"), None) \
                or next((e[x] for e in tout if e.get(x, ("",))[0] == "total"), None)
            if total:                                                         # rule 1
                totals[x] = total[1:]
                for p in range(4):
                    if tout[p].get(x, ("",))[0] != "total":
                        acts.setdefault(p, []).append(f"{x} total {total[1]} {total[2]}")
                continue
            mine = [p for p in range(4) if x in tout[p]]
            if root:
                if not mine:
                    for p in range(4):
                        acts.setdefault(p, []).append(f"{x} wave")
                elif all(x in e for e in tin):
                    subs = [e[x] for e in tin if e[x][0] == "sub"]
                    s = "+".join([str(number)] + [str(e[1]) for e in subs])
                    c = "+".join(["1"] + [str(e[2]) for e in subs])
                    for p in range(4):
                        acts.setdefault(p, []).append(f"{x} total {s} {c}")
                    totals[x] = (sim.plus(s), sim.plus(c))
                continue
            if not mine:                                                      # rule 2
                parent = next((p for p in range(4) if tin[p].get(x, ("",))[0] == "wave"), None)
                if parent is not None:
                    for p in range(4):
                        if p != parent:
                            acts.setdefault(p, []).append(f"{x} wave")
                continue
            waved = [p for p in range(4) if tout[p].get(x, ("",))[0] == "wave"]
            if len(waved) == 3 and len(mine) == 3 and all(x in tin[p] for p in waved):      # rule 3
                parent = next(p for p in range(4) if p not in waved)
                subs = [tin[p][x] for p in waved if tin[p][x][0] == "sub"]
                s = "+".join([str(number)] + [str(e[1]) for e in subs])
                c = "+".join(["1"] + [str(e[2]) for e in subs])
                acts.setdefault(parent, []).append(f"{x} sub {s} {c}")
        answer = ""
        if totals and len(totals) == len(letters) and len(set(totals.values())) == 1:
            s, c = next(iter(totals.values()))
            answer = f"total {s} over {c} processes" + (" (trees A and B agree)" if len(letters) == 2 else "")
        elif totals and len(totals) == len(letters):
            answer = "sum unknown: trees A and B disagree"
        return [f'<write port="q{p}" as="tree">{"; ".join(v)}</write>' for p, v in sorted(acts.items())], answer

    # ---------------------------------------------------------------- ring

    def ring(self, turn, me, number, ins, outs, notes):
        acts, answer = [], ""
        base = (number, 1)
        for d in range(4):
            arrived = entries(ins[d], "ring")
            kept = dict(re.findall(r"ring (\d) = (\d+ \d+)", turn.prompt.split("your ring results", 1)[1].split("\narrived on", 1)[0])) \
                if "your ring results" in turn.prompt else {}
            result = tuple(map(int, kept[str(d)].split())) if str(d) in kept else None      # rule 1: the host keeps it
            want = [f"{i} {s}+{base[0]} {c}+{base[1]}" for i, s, c in arrived if i != me]      # rule 2
            if len(want) < 2 and not result:
                want.append(f"{me} {base[0]} {base[1]}")
            packed = sim.ring_text(sim.ring_to_port("; ".join(want))[2]) if want else "none"
            if outs[d] != "ring " + packed and not (packed == "none" and outs[d] == "empty"):
                acts.append(f'<write port="q{d}" as="ring">' + ("; ".join(want) or "none") + "</write>")
            if not result:
                break
            base = result
            if d == 3:
                answer = f"total {result[0]} over {result[1]} processes"
        return acts, answer
