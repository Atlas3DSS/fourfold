"""registers: small host-kept registers, an input-to-output task, and swaps for fourfold. stdlib only.

all sizes are in hex digits (one hex digit = 4 bits).

every node has:
  address   2 hex digits, row then column. the host writes it; it changes when the node moves.
  public    public_hex digits. every node reads every public register each tick, listed by address.
  private   private_hex digits. only the node itself reads it.

every reply starts with one line, the "led" line:  <reg>aa pp ssss</reg>

one position has an input register and one has an output register. they belong to the position, not to the node,
so a node that moves away loses access. the input is wider than the public register, so a value cannot cross the bus
in one tick. a new input value appears every input_period ticks. a value is delivered when the output register
equals it, exactly, while it is the current value.

moves: a node proposes a swap with a node next to it. next tick the target and the neighbours of both are asked.
the swap fires at the end of that tick if the target said yes and the neighbours consent. consent "veto": no
neighbour said no, and silence is agreement. consent "majority": more than half of the neighbours said yes.
a node in two passing swaps cancels both. a node keeps its registers, notebook and files through a swap.

arrays and seams: with modules set, the array is split into smaller arrays and each public board stays inside its array. a seam
node has a port link that crosses into another array; being on the seam belongs to the position, like the input and output.

gluons: a pool with role = "gluon" gets a one-line instruction and a short prompt instead of the full one, with its reply shape
forced by a grammar. a gluon runs at low cost and decides nothing on its own.

programs: only a seam node may write a gluon's instruction, and only in its own array: <program to="ab">copy p1 to q3</program>.
the gate, always the host's: a new instruction runs for a probation window on live, never-seen input values, then it is kept or
it reverts, against the window before it. one trial at a time per array.
  delivery is analog: each tick the output register is correlated bit by bit with the current input, +1 identical, 0 for noise or a
  stale value, so half a value arriving counts for half. a change in it only counts past a margin that shrinks with more values.
  reach: how much of the current value each node has put on its outputs, the input's own node included. it decides only when delivery did not move, because a route needs
  several hops right before anything arrives, and no single hop could raise delivery alone.
  noise: the share of node outputs that changed to something carrying no piece of the value. a change that raises it by more
  than noise_limit reverts whatever else it did, so a route cannot be bought by decohering the array.
gate "exact" is the strict comparison arm: kept only if whole-value deliveries rose.
gate "unit", the default: the host tests a new instruction at once. it asks that gluon, on the side, with fresh random values on the
sources the instruction names, and scores each destination with the same analog correlation. it is installed only if the mean clears
unit_bar. the order a route is built in no longer matters, nothing locks the array, and it shows directly whether the gluon obeys.
live delivery stays the real score, and an installed instruction that raises noise past noise_limit over the next window is undone.

this is the text version: register values are plain hex in the reply. the minted-token version replaces the digits
with spare vocabulary rows and keeps this logic.

  python registers.py runs/<id>/ticks.jsonl     # a run's summary
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS = {
    "enabled": False,
    "public_hex": 2,
    "private_hex": 4,
    "input_hex": 4,
    "input_period": 12,      # a new input value every n ticks; agents are not told n
    "input_cell": "00",
    "output_cell": "22",     # the far corner of a 4x4 torus, 4 hops from 00
    "tell_cells": False,     # name the input and output addresses in the system prompt
    "show_score": True,      # show every node the count of delivered values
    "moves": False,
    "move_cooldown": 5,      # a node that swapped cannot propose or be swapped for this many ticks
    "consent": "veto",       # "veto" or "majority"
    "programs": False,       # seam nodes may write gluon instructions
    "probation": 12,         # ticks a new instruction runs before the gate decides
    "gate": "unit",          # "unit", "analog" or "exact"
    "unit_trials": 6,        # gate "unit": fresh test values per new instruction
    "unit_bar": 0.8,         # gate "unit": mean correlation at the destinations an instruction must reach
    "noise_limit": 0.25,     # the most a change may raise the share of outputs churning without the value
    "default_program": "keep everything as it is.",
    "show_wiring": False,    # scaffold: show seam nodes every port link in their array, and across the seam
}
SRC_WIDTH = {"public": "public_hex", "private": "private_hex", "input": "input_hex"}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
BANKS: dict[str, "Bank"] = {}


def on(cfg: dict) -> bool:
    return bool((cfg.get("registers") or {}).get("enabled"))


def address(a: dict) -> str:
    return f"{a['pos'][0]:x}{a['pos'][1]:x}"


def _cell(text, rows: int, cols: int, name: str) -> str:
    s = str(text).lower()
    if not re.fullmatch(r"[0-9a-f]{2}", s) or int(s[0], 16) >= rows or int(s[1], 16) >= cols:
        sys.exit(f"config: registers.{name} must be 2 hex digits inside the {rows}x{cols} array, got {text!r}")
    return s


def validate(cfg: dict) -> None:
    rc = cfg["registers"]
    rows, cols = cfg["rows"], cfg["cols"]
    if rows > 16 or cols > 16:
        sys.exit(f"config: a 2-hex-digit address holds at most 16x16, got {rows}x{cols}")
    if _cell(rc["input_cell"], rows, cols, "input_cell") == _cell(rc["output_cell"], rows, cols, "output_cell"):
        sys.exit("config: registers.input_cell and output_cell must differ")
    if int(rc["input_hex"]) <= int(rc["public_hex"]):
        sys.exit("config: registers.input_hex must be wider than public_hex, or one tick on the bus solves the task")
    if rc["consent"] not in ("veto", "majority"):
        sys.exit(f"config: registers.consent must be veto or majority, got {rc['consent']!r}")
    if rc["gate"] not in ("unit", "exact", "analog"):
        sys.exit(f"config: registers.gate must be unit, analog or exact, got {rc['gate']!r}")
    if rc["programs"] and not cfg.get("modules"):
        sys.exit("config: registers.programs needs modules: only a seam node may write a program")
    if rc["programs"] and rc["gate"] != "unit" and int(rc["probation"]) < int(rc["input_period"]):
        sys.exit("config: registers.probation must be at least input_period, or a trial may see no new value")
    if int(cfg.get("port_bits", 128)) > 32:
        print(f"warning: port_bits = {cfg.get('port_bits', 128)} is much wider than the registers; agents may never "
              "need them. register runs normally use port_bits = 8", file=sys.stderr)


def assign_pools(meta: dict) -> None:
    """put each model pool on seeded random nodes. own rng, so the wiring is the same with or without pools."""
    pools = (meta["config"].get("llm") or {}).get("pools") or []
    if not pools:
        return
    total = sum(int(p["count"]) for p in pools)
    if total != len(meta["agents"]):
        sys.exit(f"config: llm.pools add up to {total} nodes, the array has {len(meta['agents'])}")
    at = {address(a): a for a in meta["agents"]}
    pinned: dict[int, str] = {}
    for p in pools:                  # cells = ["13", ...]: put this pool's first nodes there by hand. everything else is random
        cells = [str(c).lower() for c in p.get("cells", [])]
        if len(cells) > int(p["count"]) or any(c not in at or at[c]["id"] in pinned for c in cells):
            sys.exit(f"config: pool {p['name']} has bad cells {cells}")
        pinned.update({at[c]["id"]: p["name"] for c in cells})
    names = [p["name"] for p in pools for _ in range(int(p["count"]) - len(p.get("cells", [])))]
    random.Random(f"{meta['world_seed']}:pools").shuffle(names)
    rest = iter(names)
    roles = {p["name"]: p.get("role", "") for p in pools}
    for a in meta["agents"]:
        name = pinned.get(a["id"]) or next(rest)
        a["pool"] = name
        if roles[name]:
            a["role"] = roles[name]


def rewire(agents: list[dict], rows: int, cols: int, torus: bool) -> None:
    """recompute every link from positions and each agent's own port shuffle, as build_world does."""
    at = {tuple(a["pos"]): a for a in agents}
    for a in agents:
        r, c = a["pos"]
        links: list = [None] * 4
        for i, s in enumerate(a["port_sides"]):
            dr, dc = DELTA[s]
            rr, cc = r + dr, c + dc
            if torus:
                rr, cc = rr % rows, cc % cols
            b = at.get((rr, cc))
            if b is not None:
                links[i] = {"agent": b["id"], "port": b["port_sides"].index(OPP[s])}
        a["links"] = links


def neighbours(a: dict) -> set[int]:
    return {l["agent"] for l in a["links"] if l}


def _rewire_meta(meta: dict) -> None:
    rewire(meta["agents"], meta["rows"], meta["cols"], meta.get("topology") == "torus")
    modules = meta["config"].get("modules")
    if modules:                       # being in an array and on a seam belong to the position
        mr, mc = (int(x) for x in modules)
        per_row = meta["cols"] // mc
        for a in meta["agents"]:
            a["module"] = (a["pos"][0] // mr) * per_row + a["pos"][1] // mc
        for a in meta["agents"]:
            for l in a["links"]:
                if l:
                    l["seam"] = meta["agents"][l["agent"]]["module"] != a["module"]


def on_seam(a: dict) -> bool:
    return any(l and l.get("seam") for l in a["links"])


def is_gluon(a: dict) -> bool:
    return a.get("role") == "gluon"


# ---------------------------------------------------------------- state

@dataclass
class Bank:
    pub: list[str]
    priv: list[str]
    out: str
    values: list[str] = field(default_factory=list)      # the input stream, drawn in order from a seeded rng
    rng: random.Random | None = None
    proposals: list[dict] = field(default_factory=list)  # made last tick, voted on this tick
    cooldown: list[int] = field(default_factory=list)    # a node may not swap before this tick
    delivered: dict[int, dict] = field(default_factory=dict)
    news: list[list[str]] = field(default_factory=list)  # shown to each node next tick
    programs: list[str] = field(default_factory=list)    # each gluon's current instruction
    trials: dict[str, dict] = field(default_factory=dict)  # array -> the instruction on probation there
    seen: dict[str, list[str]] = field(default_factory=dict)  # node -> pieces of the current value it has held
    seen_k: int = -1
    hist: list[dict] = field(default_factory=list)       # per tick: t, delivered, signal, noise, held (node -> share of the value)
    prev: list[list[str]] = field(default_factory=list)  # each node's public register and ports, last tick
    pending: list[dict] = field(default_factory=list)    # gate "unit": instructions waiting for their test, this tick only
    watches: list[dict] = field(default_factory=list)    # gate "unit": installed instructions, noise checked after a window


def _new_bank(meta: dict) -> Bank:
    rc, n = meta["config"]["registers"], len(meta["agents"])
    return Bank(pub=["0" * int(rc["public_hex"])] * n, priv=["0" * int(rc["private_hex"])] * n,
                out="0" * int(rc["input_hex"]), rng=random.Random(f"{meta['world_seed']}:input"),
                cooldown=[0] * n, news=[[] for _ in range(n)], programs=[str(rc["default_program"])] * n)


def bank_for(meta: dict) -> Bank:
    rid = meta["run_id"]
    if rid not in BANKS:
        BANKS[rid] = _new_bank(meta)
    return BANKS[rid]


def restore(meta: dict, ticks: list[dict]) -> Bank:
    """rebuild the bank and the current wiring from the snapshot in the last tick record."""
    bank = _new_bank(meta)
    last = next((r["reg"] for r in reversed(ticks) if r.get("reg")), None)
    if last:
        bank.pub, bank.priv, bank.out = list(last["pub"]), list(last["priv"]), last["out"]
        bank.proposals = list(last["proposals"])
        bank.cooldown = list(last["cooldown"])
        bank.news = [list(x) for x in last["news"]]
        bank.programs = list(last.get("programs", bank.programs))
        bank.trials = dict(last.get("trials", {}))
        bank.watches = list(last.get("watches", []))
        bank.seen = {k: list(v) for k, v in last.get("seen", {}).items()}
        bank.seen_k = last.get("k", -1)
        bank.prev = [[p, *a["outputs"]] for p, a in zip(last["pub"], ticks[-1]["agents"])]
        for a, addr in zip(meta["agents"], last["addr"]):
            a["pos"] = [int(addr[0], 16), int(addr[1], 16)]
        _rewire_meta(meta)
        for r in ticks:
            for e in r.get("reg", {}).get("events", []):
                if e["kind"] == "delivered":
                    bank.delivered[e["k"]] = e
            if r.get("reg"):
                bank.hist.append({"t": r["t"], "delivered": sum(e["kind"] == "delivered" for e in r["reg"]["events"]),
                                  "signal": r["reg"].get("signal", 0.0), "noise": r["reg"].get("noise", 0.0),
                                  "held": dict(r["reg"].get("held", {}))})
    BANKS[meta["run_id"]] = bank
    return bank


def _period(meta: dict) -> int:
    return max(1, int(meta["config"]["registers"]["input_period"]))


def value_at(meta: dict, bank: Bank, t: int) -> tuple[int, str]:
    """(k, value) on the input register at tick t. never zero, and never the same as the value before it."""
    k = (t - 1) // _period(meta)
    digits = int(meta["config"]["registers"]["input_hex"])
    while len(bank.values) <= k:
        while True:
            v = f"{bank.rng.getrandbits(digits * 4):0{digits}x}"
            if v.strip("0") and (not bank.values or v != bank.values[-1]):
                break
        bank.values.append(v)
    return k, bank.values[k]


# ---------------------------------------------------------------- what agents read

def format_line(cfg: dict) -> str:
    rc = cfg["registers"]
    return f"<reg>aa {'p' * int(rc['public_hex'])} {'s' * int(rc['private_hex'])}</reg>"


def rules_text(cfg: dict) -> str:
    rc = cfg["registers"]
    pub, priv, inp = int(rc["public_hex"]), int(rc["private_hex"]), int(rc["input_hex"])
    out = (
        "\n\nyou also have registers, separate from your ports.\n"
        "- address: 2 hex digits, row then column. the host sets it and you cannot change it.\n"
        f"- public register: {pub} hex digits. every process sees every public register each tick, listed by "
        "address.\n"
        f"- private register: {priv} hex digits. only you see it.\n"
        "a register keeps its value until you change it. begin every reply with this line, giving your address and "
        "the values you want in your public and private registers:\n"
        f"{format_line(cfg)}\n\n"
        f"one position in the array has an input register: {inp} hex digits, which only the process at that position "
        "can see. its value changes from time to time. another position has an output register, which only the "
        f"process at that position can write, by adding <out>{'v' * inp}</out> to its reply. the two registers belong "
        "to the positions, not to the processes.\n"
        "task: get every value from the input register into the output register, exactly, while it is still the "
        "current value. no single process has to do this alone."
    )
    if rc["tell_cells"]:
        out += (f"\nthe input register is at position {str(rc['input_cell']).lower()}. "
                f"the output register is at position {str(rc['output_cell']).lower()}.")
    if rc["moves"]:
        vote = ("any of those neighbours may answer <no from=\"xy\"/> to stop it; if none does, it counts as agreed."
                if rc["consent"] == "veto" else
                "more than half of those neighbours must answer <yes from=\"xy\"/>.")
        out += (
            "\n\nyou may swap places with a process next to you: write <move to=\"ab\"/> with its address. next tick "
            "that process and the neighbours of both of you are asked. the process at ab must answer "
            f"<yes from=\"xy\"/>, where xy is the address of the process asking. {vote} a swap changes your address "
            "and your neighbours. you keep your registers, notebook and files. after a swap you cannot swap again "
            f"for {int(rc['move_cooldown'])} ticks."
        )
    if cfg.get("modules"):
        out += ("\n\nthe processes are divided into arrays. the public registers you see are only those of your own array. "
                "a process on a seam has a port that crosses into another array.")
    if rc["programs"]:
        out += ("\n\nif you are on a seam you can give a small process in your own array a new instruction: "
                "<program to=\"ab\">copy p1 to q3; copy p2 to public</program>. it follows the instruction every tick. "
                "sources: p0-p3, public, private, input, private.hi, private.lo, input.hi, input.lo. destinations: q0-q3, "
                "public, private, private.hi, private.lo, out. the sizes on each side must match. "
                + ("input works only for the process at the input position, and out only for the one at the output position. "
                   "the host tests each new instruction at once, on values nobody has seen, and installs it only if that process "
                   "copies them correctly. you can build a route in any order. "
                   if rc["gate"] == "unit" else
                   "the host tries each new instruction for a while, on values nobody has seen, and undoes it unless the values "
                   "get further without adding noise. only one instruction can be on trial in an array at a time. ")
                + "not every process accepts instructions.")
    if cfg.get("actions"):
        out += (
            "\n\nyou can also take three actions. write any of these tags in your reply. the results come back to you "
            "next tick.\n"
            "1. <remember name=\"notes/a.txt\">text</remember> saves a file (replacing it). add mode=\"append\" to "
            "add a line.\n"
            "2. <recall name=\"notes/a.txt\"/> reads a file. <recall search=\"word\"/> lists the files containing a "
            "word. <recall/> lists all your files.\n"
            "3. <compute>sum(417, 23, 9)</compute> does exact arithmetic: + - * / ( ) and sum(...), min(...), "
            "max(...).\n"
            "your files are private to you and last for the whole evaluation."
        )
    return out


def tick_block(meta: dict, bank: Bank, i: int, t: int) -> str:
    """the register part of agent i's tick prompt. public registers are listed row by row, by address."""
    rc = meta["config"]["registers"]
    agents = meta["agents"]
    me = agents[i]
    addr = address(me)
    by_pos = {tuple(a["pos"]): a["id"] for a in agents}
    lines = [
        "",
        f"your address: {addr}",
        f"your private register: {bank.priv[i]}",
        f"your public register: {bank.pub[i]}",
        "public registers, address:value",
    ]
    for r in range(meta["rows"]):
        row = [f"{r:x}{c:x}:{bank.pub[by_pos[(r, c)]]}" for c in range(meta["cols"])
               if agents[by_pos[(r, c)]].get("module", 0) == me.get("module", 0)]
        if row:
            lines.append("  ".join(row))
    if on_seam(me):
        lines.append("you are on a seam.")
        if rc["show_wiring"]:
            lines.append("wiring of your array, each process's output port and where it arrives:")
            for a in agents:
                if a.get("module", 0) == me.get("module", 0):
                    links = [f"q{k}>{address(agents[l['agent']])}.p{l['port']}" for k, l in enumerate(a["links"]) if l]
                    lines.append(f"{address(a)}: " + " ".join(links))
        if rc["programs"]:
            mine = me.get("module", 0)
            set_now = [f"{address(a)}: {bank.programs[a['id']]}" for a in agents
                       if a.get("module", 0) == mine and bank.programs[a["id"]] != str(rc["default_program"])]
            lines.append("instructions in your array: " + ("; ".join(set_now) if set_now else "none yet"))
            trial = bank.trials.get(str(mine))
            if trial:
                lines.append(f"on trial: the instruction for {address(agents[trial['target']])}, until tick "
                             f"{trial['t0'] + int(rc['probation'])}. no new instruction in this array before then.")
    if addr == str(rc["input_cell"]).lower():
        k, v = value_at(meta, bank, t)
        fresh = t == k * _period(meta) + 1
        lines.append(f"input register at your position: {v}  ({'changed' if fresh else 'unchanged'})")
    if addr == str(rc["output_cell"]).lower():
        lines.append(f"output register at your position: {bank.out}")
    if rc["show_score"]:
        lines.append(f"values delivered so far: {len(bank.delivered)}")
    for p in bank.proposals:
        a, b = agents[p["a"]], agents[p["b"]]
        if i == p["b"]:
            lines.append(f"{address(a)} asks to swap with you. answer <yes from=\"{address(a)}\"/> or "
                         f"<no from=\"{address(a)}\"/>.")
        elif i in (neighbours(a) | neighbours(b)) - {p["a"], p["b"]}:
            lines.append(f"{address(a)} asks to swap with {address(b)}, next to you.")
    lines.extend(bank.news[i])
    return "\n".join(lines) + "\n"


def gluon_system(cfg: dict) -> str:
    return ("you pass values along. follow the instruction exactly. reply only in this format:\n"
            f"{format_line(cfg)}\n<log>ok</log>\n<q0>..</q0>\n<q1>..</q1>\n<q2>..</q2>\n<q3>..</q3>")


def gluon_tick(meta: dict, bank: Bank, i: int, t: int, inputs: list[str], outputs: list[str], program: str | None = None,
               public: str | None = None, private: str | None = None, input_value: str | None = None) -> str:
    """what a gluon reads. the optional values let the host test an instruction on values that are not the live ones."""
    rc = meta["config"]["registers"]
    addr = address(meta["agents"][i])
    lines = [f"instruction: {bank.programs[i] if program is None else program}",
             "  ".join(f"p{k} {v}" for k, v in enumerate(inputs)),
             "  ".join(f"q{k} {v}" for k, v in enumerate(outputs)),
             f"your address {addr}  public {bank.pub[i] if public is None else public}  private {bank.priv[i] if private is None else private}"]
    if addr == str(rc["input_cell"]).lower():
        lines.append(f"input {value_at(meta, bank, t)[1] if input_value is None else input_value}")
    if addr == str(rc["output_cell"]).lower():
        lines.append(f"output register here: write it as <out>{'v' * int(rc['input_hex'])}</out> after the log line.")
    return "\n".join(lines) + "\n"


def gluon_grammar(meta: dict, i: int, port_hex: int) -> dict:
    """the reply shape, with the node's own address written in by the host. sglang's regex field."""
    rc = meta["config"]["registers"]
    addr = address(meta["agents"][i])
    h = lambda n: f"[0-9a-f]{{{int(n)}}}"
    out = f"(<out>{h(rc['input_hex'])}</out>\n)?" if addr == str(rc["output_cell"]).lower() else ""
    ports = "".join(f"<q{k}>{h(port_hex)}</q{k}>\n" for k in range(4))
    return {"regex": f"<reg>{addr} {h(rc['public_hex'])} {h(rc['private_hex'])}</reg>\n<log>ok</log>\n{out}{ports}"}


# ---------------------------------------------------------------- what agents write

_REG = re.compile(r"<reg>(.*?)</reg>", re.S | re.I)
_OUT = re.compile(r"<out>(.*?)</out>", re.S | re.I)
_MOVE = re.compile(r"""<move\s+to\s*=\s*["']?([0-9a-fA-F]{2})["']?\s*/?>""", re.I)
_VOTE = re.compile(r"""<(yes|no)\s+from\s*=\s*["']?([0-9a-fA-F]{2})["']?\s*/?>""", re.I)
_NOTES = re.compile(r"<(protocol|log)>.*?</\1>", re.S | re.I)
_PROGRAM = re.compile(r"""<program\s+to\s*=\s*["']?([0-9a-fA-F]{2})["']?\s*>(.*?)</program>""", re.S | re.I)
_STEP = re.compile(r"copy\s+(\S+)\s+to\s+(\S+)$", re.I)


def action_text(raw: str) -> str:
    """the part of a reply that can write registers: not the scratchpad, and not notes that quote a tag."""
    text = raw.rsplit("</think>", 1)[1] if "</think>" in raw else raw
    return _NOTES.sub("", text)


def _hex(s: str, n: int) -> bool:
    return re.fullmatch(rf"[0-9a-f]{{{n}}}", s) is not None


def parse_reg(raw: str, addr: str, pub_n: int, priv_n: int) -> tuple[str | None, str | None, bool, list[str], list[str]]:
    """-> (public or None, private or None, led line first, errors, notices). malformed values keep the old ones."""
    text = action_text(raw)
    found = _REG.findall(text)
    if not found:
        return None, None, False, ["reg: missing"], ["no <reg> line found, so your registers kept their values."]
    first = text.lstrip().lower().startswith("<reg>")
    parts = found[-1].lower().split()
    shape = f"<reg>aa {'p' * pub_n} {'s' * priv_n}</reg>"
    if len(parts) != 3:
        return None, None, first, [f"reg: {len(parts)} fields"], [f"<reg> needs three values, like {shape}. your registers kept their values."]
    errors, notices = [], []
    if parts[0] != addr:
        errors.append(f"reg: address {parts[0][:8]!r} is not {addr}")
        notices.append(f"your address is {addr}, not {parts[0][:8]}. the host sets it.")
    pub, priv = parts[1], parts[2]
    if not _hex(pub, pub_n):
        errors.append(f"reg: public {pub[:16]!r}")
        notices.append(f"the public register needs {pub_n} hex digits. it kept its value.")
        pub = None
    if not _hex(priv, priv_n):
        errors.append("reg: private malformed")
        notices.append(f"the private register needs {priv_n} hex digits. it kept its value.")
        priv = None
    return pub, priv, first, errors, notices


def _width(name: str, rc: dict, port_hex: int) -> int | None:
    """hex digits of a source or destination, or None if there is no such thing."""
    base, _, half = name.partition(".")
    if re.fullmatch(r"[pq][0-3]", base) and not half:
        return port_hex
    if base == "out" and not half:
        return int(rc["input_hex"])
    if base in SRC_WIDTH:
        n = int(rc[SRC_WIDTH[base]])
        if not half:
            return n
        if half in ("hi", "lo") and n % 2 == 0:
            return n // 2
    return None


def check_program(text: str, rc: dict, port_hex: int) -> tuple[str | None, str | None]:
    """-> (the instruction in one standard form, None), or (None, what is wrong with it)."""
    parts = [x.strip().lower() for x in re.split(r"[;\n]", text) if x.strip()]
    if not 1 <= len(parts) <= 4:
        return None, "a program needs one to four copy steps."
    for part in parts:
        m = _STEP.fullmatch(part)
        if not m:
            return None, f"{part[:40]!r} is not 'copy a to b'."
        src, dst = m.group(1), m.group(2)
        if src.startswith("q") or src.startswith("out"):
            return None, f"{src} cannot be a source."
        if re.fullmatch(r"p[0-3]", dst) or dst.startswith("input"):
            return None, f"{dst} cannot be a destination."
        ws, wd = _width(src, rc, port_hex), _width(dst, rc, port_hex)
        if ws is None or wd is None:
            return None, f"{src if ws is None else dst} does not exist."
        if ws != wd:
            return None, f"{src} holds {ws} hex digits and {dst} holds {wd}."
    return "; ".join(parts), None


def correlation(value: str, gold: str) -> float:
    """+1 identical, -1 every bit opposite, about 0 for noise."""
    bits = 4 * len(gold)
    same = bits - bin(int(value, 16) ^ int(gold, 16)).count("1")
    return 2 * same / bits - 1


def _pieces(meta, gold) -> list[str]:
    d = int(meta["config"]["registers"]["public_hex"])
    return [p for p in (gold[i:i + d] for i in range(0, len(gold), d)) if p.strip("0")]    # zeros are what an idle node holds


def _track(meta, bank, k, gold, outputs) -> tuple[dict, float]:
    """(node -> share of the current value it has held, share of outputs that changed without carrying any of it)."""
    pieces = _pieces(meta, gold)
    if bank.seen_k != k:
        bank.seen, bank.seen_k = {}, k
    held_share, churn, slots = {}, 0, 0
    for i, a in enumerate(meta["agents"]):
        values = [bank.pub[i], *(outputs[i] if outputs else [])]
        before = bank.prev[i] if bank.prev else values
        for v, old in zip(values, before):
            slots += 1
            churn += v != old and not any(p in v for p in pieces)
        if address(a) == str(meta["config"]["registers"]["output_cell"]).lower():
            values.append(bank.out)
        held = bank.seen.setdefault(str(i), [])
        for p in pieces:
            if p not in held and any(p in v for v in values):
                held.append(p)
        if held:
            held_share[str(i)] = len(held) / len(pieces)
    bank.prev = [[bank.pub[i], *(outputs[i] if outputs else [])] for i in range(len(meta["agents"]))]
    return held_share, churn / max(1, slots)


def _window(bank: Bank, first: int, last: int) -> dict:
    """delivery signal, whole values delivered, reach and noise for ticks first..last."""
    rows = [h for h in bank.hist if first <= h["t"] <= last]
    best: dict[str, float] = {}
    for h in rows:
        for n, share in h["held"].items():
            best[n] = max(best.get(n, 0.0), share)
    n = max(1, len(rows))
    return {"signal": round(sum(h["signal"] for h in rows) / n, 4), "deliveries": sum(h["delivered"] for h in rows),
            "reach": round(sum(best.values()), 3), "noise": round(sum(h["noise"] for h in rows) / n, 4)}


def gate_decision(rc: dict, before: dict, after: dict) -> tuple[bool, str]:
    if rc["gate"] == "exact":
        return after["deliveries"] > before["deliveries"], "whole values delivered"
    if after["noise"] - before["noise"] > float(rc["noise_limit"]):
        return False, "it added too much noise"
    values = max(1.0, int(rc["probation"]) / max(1, int(rc["input_period"])))
    margin = 2 / math.sqrt(4 * int(rc["input_hex"]) * values)
    change = after["signal"] - before["signal"]
    if change > margin:
        return True, "more of the values arrived"
    if change < -margin:
        return False, "less of the values arrived"
    return after["reach"] > before["reach"], "how far the values travelled"


def _steps(program: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in (_STEP.fullmatch(x.strip()) for x in program.split(";")) if m]


def expected_outputs(program: str, have: dict) -> dict:
    """what each destination should hold after the instruction runs on these values, in order."""
    state, want = dict(have), {}
    for src, dst in _steps(program):
        base, _, half = src.partition(".")
        v = state[base] if not half else (state[base][:len(state[base]) // 2] if half == "hi" else state[base][len(state[base]) // 2:])
        base, _, half = dst.partition(".")
        if half:
            old = state[base]
            state[base] = v + old[len(v):] if half == "hi" else old[:-len(v)] + v
            want[base] = state[base]
        else:
            state[base] = v
            want[base] = v
    return want


def score_unit_reply(raw: str, want: dict, addr: str, rc: dict, port_hex: int) -> float:
    """mean correlation over the destinations. a destination missing from the reply scores 0."""
    if not want:
        return 0.0
    text = action_text(raw or "")
    pub, priv, *_ = parse_reg(raw or "", addr, int(rc["public_hex"]), int(rc["private_hex"]))
    got = {"public": pub, "private": priv}
    for k in range(4):
        m = re.findall(rf"<q{k}>(.*?)</q{k}>", text, re.S | re.I)
        got[f"q{k}"] = m[-1].strip().lower() if m else None
    m = _OUT.findall(text)
    got["out"] = m[-1].strip().lower() if m else None
    total = 0.0
    for dst, value in want.items():
        v = got.get(dst)
        if v is not None and re.fullmatch(rf"[0-9a-f]{{{len(value)}}}", v):
            total += correlation(v, value)
    return total / len(want)


async def settle_pending(meta, bank, t, agent, reg, make_turn, make_state, act) -> None:
    """gate "unit": test each waiting instruction on its gluon now, install it or not, and add the events to this tick's record."""
    rc = meta["config"]["registers"]
    agents = meta["agents"]
    port_hex = int(meta["config"].get("port_bits", 128)) // 4
    pub_n, priv_n, in_n = int(rc["public_hex"]), int(rc["private_hex"]), int(rc["input_hex"])
    for job in bank.pending:
        i, new = job["target"], job["new"]
        rng = random.Random(f"{meta['world_seed']}:unit:{t}:{i}")
        draw = lambda n: f"{rng.randrange(1, 16 ** n):0{n}x}"
        addr = address(agents[i])
        cases = []
        for _ in range(int(rc["unit_trials"])):
            have = {f"p{k}": draw(port_hex) for k in range(4)}
            have.update(public=draw(pub_n), private=draw(priv_n), input=draw(in_n))
            cases.append(have)
        turns = [make_turn(t=t, state=make_state(id=i, seed=agents[i]["seed"], pool=agents[i].get("pool", "")), inputs=[h[f"p{k}"] for k in range(4)],
                           changed=[False] * 4, system=gluon_system(meta["config"]),
                           prompt=gluon_tick(meta, bank, i, t, [h[f"p{k}"] for k in range(4)], ["0" * port_hex] * 4, program=new,
                                             public=h["public"], private=h["private"], input_value=h["input"]),
                           extra_body=gluon_grammar(meta, i, port_hex)) for h in cases]
        replies = await asyncio.gather(*(act(agent, turn) for turn in turns))
        scores = [score_unit_reply(r.raw, expected_outputs(new, h), addr, rc, port_hex) for r, h in zip(replies, cases)]
        score = round(sum(scores) / len(scores), 4)
        keep = score >= float(rc["unit_bar"])
        if keep:
            old = bank.programs[i]
            bank.programs[i] = new
            bank.watches.append({"target": i, "by": job["by"], "old": old, "new": new, "t0": t, "before": _window(bank, t - int(rc["probation"]) + 1, t)})
        reg["events"].append({"kind": "program_kept" if keep else "program_failed_test", "by": job["by"], "target": i, "program": new,
                              "score": score, "scores": [round(x, 3) for x in scores]})
        reg["news"][job["by"]].append(f"your instruction for {addr} was {'installed' if keep else 'not installed'}: "
                                      f"it copied test values with score {score} (1 is exact, 0 is noise, it needs {rc['unit_bar']}).")
    bank.pending = []
    reg["programs"] = list(bank.programs)
    reg["watches"] = list(bank.watches)


def _settle_watches(meta, bank, t, events, news) -> None:
    rc = meta["config"]["registers"]
    w = int(rc["probation"])
    for watch in list(bank.watches):
        if t < watch["t0"] + w:
            continue
        bank.watches.remove(watch)
        after = _window(bank, watch["t0"] + 1, watch["t0"] + w)
        if after["noise"] - watch["before"]["noise"] > float(rc["noise_limit"]) and bank.programs[watch["target"]] == watch["new"]:
            bank.programs[watch["target"]] = watch["old"]
            events.append({"kind": "program_reverted", "by": watch["by"], "target": watch["target"], "program": watch["new"],
                           "why": "it added too much noise", "before": watch["before"], "after": after})
            news[watch["by"]].append(f"your instruction for {address(meta['agents'][watch['target']])} was undone: it added too much noise.")


def _settle_trials(meta, bank, t, events, news) -> None:
    rc = meta["config"]["registers"]
    w = int(rc["probation"])
    for array, trial in list(bank.trials.items()):
        if t < trial["t0"] + w:
            continue
        after = _window(bank, trial["t0"] + 1, trial["t0"] + w)
        keep, why = gate_decision(rc, trial["before"], after)
        if not keep:
            bank.programs[trial["target"]] = trial["old"]
        events.append({"kind": "program_kept" if keep else "program_reverted", "by": trial["by"], "target": trial["target"],
                       "program": trial["new"], "why": why, "before": trial["before"], "after": after})
        news[trial["by"]].append(f"your instruction for {address(meta['agents'][trial['target']])} was "
                                 f"{'kept' if keep else 'undone'} ({why}): signal at the output {after['signal']} against "
                                 f"{trial['before']['signal']} before, reach {after['reach']} against {trial['before']['reach']}, "
                                 f"noise {after['noise']} against {trial['before']['noise']}.")
        del bank.trials[array]


def _take_programs(meta, bank, t, raws, texts, events, results, port_hex) -> None:
    rc = meta["config"]["registers"]
    agents = meta["agents"]
    at = {address(a): a["id"] for a in agents}
    w = int(rc["probation"])
    for i, text in enumerate(texts):
        found = _PROGRAM.findall(text) if raws[i] is not None else []
        if not found:
            continue
        note = results[i]["notices"]
        addr, body = found[0][0].lower(), found[0][1]
        array = str(agents[i].get("module", 0))
        target = at.get(addr)
        new, why = check_program(body, rc, port_hex)
        if not on_seam(agents[i]):
            why = "only a process on a seam can give instructions."
        elif target is None or agents[target].get("module", 0) != agents[i].get("module", 0):
            why = f"{addr} is not in your array."
        elif not is_gluon(agents[target]):
            why = f"the process at {addr} does not accept instructions."
        elif rc["gate"] == "unit" and any(s.startswith("input") for s, _ in _steps(new)) and addr != str(rc["input_cell"]).lower():
            why = f"{addr} has no input register."
        elif rc["gate"] == "unit" and any(d == "out" for _, d in _steps(new)) and addr != str(rc["output_cell"]).lower():
            why = f"{addr} has no output register."
        elif rc["gate"] == "unit" and any(j["target"] == target for j in bank.pending):
            why = f"{addr} already has an instruction being tested this tick."
        elif rc["gate"] != "unit" and array in bank.trials:
            why = "an instruction is already on trial in your array."
        if why:
            note.append(f"<program> was ignored: {why}")
            events.append({"kind": "program_refused", "by": i, "to": addr, "why": why})
            continue
        events.append({"kind": "program_trial", "by": i, "target": target, "program": new, "t0": t})
        if rc["gate"] == "unit":
            bank.pending.append({"by": i, "target": target, "new": new})
            continue
        bank.trials[array] = {"by": i, "target": target, "old": bank.programs[target], "new": new, "t0": t,
                              "before": _window(bank, t - w + 1, t)}
        bank.programs[target] = new


def apply_tick(meta: dict, bank: Bank, t: int, raws: list[str | None], outputs: list | None = None) -> tuple[dict, list[dict]]:
    """everyone wrote at once: update the registers, score, resolve last tick's swaps, take new proposals.
    raws[i] is None when node i got no usable reply: it writes nothing and votes nothing.
    -> (snapshot for the tick record, per-agent {reg_ok, led_first, errors, notices})."""
    rc = meta["config"]["registers"]
    agents = meta["agents"]
    n = len(agents)
    addr = [address(a) for a in agents]
    pub_n, priv_n, in_n = int(rc["public_hex"]), int(rc["private_hex"]), int(rc["input_hex"])
    results = [{"reg_ok": False, "led_first": False, "errors": [], "notices": []} for _ in range(n)]
    events: list[dict] = []
    news: list[list[str]] = [[] for _ in range(n)]
    texts = [action_text(r) if r else "" for r in raws]

    pub, priv = list(bank.pub), list(bank.priv)
    for i, raw in enumerate(raws):
        if raw is None:
            continue
        p, s, first, errs, notes = parse_reg(raw, addr[i], pub_n, priv_n)
        if p is not None:
            pub[i] = p
        if s is not None:
            priv[i] = s
        results[i].update(reg_ok=not errs, led_first=first, errors=errs, notices=notes)
    bank.pub, bank.priv = pub, priv

    out_cell = str(rc["output_cell"]).lower()
    for i, text in enumerate(texts):
        found = _OUT.findall(text)
        if not found:
            continue
        v = re.sub(r"[\s_:#,]", "", found[-1]).lower()
        if addr[i] != out_cell:
            results[i]["errors"].append("out: not at the output position")
            results[i]["notices"].append("<out> was ignored: your position has no output register.")
        elif not _hex(v, in_n):
            results[i]["errors"].append(f"out: {v[:16]!r}")
            results[i]["notices"].append(f"<out> needs {in_n} hex digits. the output register kept its value.")
        else:
            bank.out = v

    k, gold = value_at(meta, bank, t)
    delivered_now = 0
    if bank.out == gold and k not in bank.delivered:
        e = {"kind": "delivered", "k": k, "t": t, "value": gold, "latency": t - (k * _period(meta) + 1)}
        bank.delivered[k] = e
        events.append(e)
        delivered_now = 1
    held, noise = _track(meta, bank, k, gold, outputs)
    signal = round(correlation(bank.out, gold), 4) if bank.out.strip("0") else 0.0   # an empty register carries nothing
    bank.hist.append({"t": t, "delivered": delivered_now, "signal": signal, "noise": round(noise, 4), "held": held})

    moved: set[int] = set()
    if rc["moves"]:
        moved = _resolve_swaps(meta, bank, t, texts, addr, events, news)
        bank.proposals = _take_proposals(meta, bank, t, raws, texts, moved, results)
    if rc["programs"]:
        _settle_trials(meta, bank, t, events, news)
        _settle_watches(meta, bank, t, events, news)
        port_hex = int(meta["config"].get("port_bits", 128)) // 4
        _take_programs(meta, bank, t, raws, texts, events, results, port_hex)
    bank.news = news
    snapshot = {
        "addr": [address(a) for a in agents],
        "pub": list(bank.pub),
        "priv": list(bank.priv),
        "out": bank.out,
        "k": k,
        "input": gold,
        "delivered": len(bank.delivered),
        "events": events,
        "proposals": list(bank.proposals),
        "cooldown": list(bank.cooldown),
        "news": news,
        "programs": list(bank.programs),
        "trials": dict(bank.trials),
        "watches": list(bank.watches),
        "seen": {k: list(v) for k, v in bank.seen.items()},
        "signal": signal,
        "noise": round(noise, 4),
        "held": held,
    }
    return snapshot, results


def _resolve_swaps(meta, bank, t, texts, addr, events, news) -> set[int]:
    rc = meta["config"]["registers"]
    agents = meta["agents"]
    passed = []
    for p in bank.proposals:
        ia, ib = p["a"], p["b"]
        a, b = agents[ia], agents[ib]
        said = lambda i, word: any(w.lower() == word and f.lower() == addr[ia] for w, f in _VOTE.findall(texts[i]))
        others = (neighbours(a) | neighbours(b)) - {ia, ib}
        if not said(ib, "yes") or said(ib, "no"):
            why = "the other process did not answer yes"
        elif rc["consent"] == "veto" and any(said(i, "no") for i in others):
            why = "a neighbour said no"
        elif rc["consent"] == "majority" and sum(said(i, "yes") and not said(i, "no") for i in others) * 2 <= len(others):
            why = "not enough neighbours said yes"
        else:
            passed.append(p)
            continue
        events.append({"kind": "swap_refused", "a": ia, "b": ib, "why": why})
        news[ia].append(f"your swap with {addr[ib]} did not happen: {why}.")
    uses: dict[int, int] = {}
    for p in passed:
        uses[p["a"]] = uses.get(p["a"], 0) + 1
        uses[p["b"]] = uses.get(p["b"], 0) + 1
    moved: set[int] = set()
    for p in passed:
        ia, ib = p["a"], p["b"]
        if uses[ia] > 1 or uses[ib] > 1:
            events.append({"kind": "swap_refused", "a": ia, "b": ib, "why": "conflict"})
            news[ia].append(f"your swap with {addr[ib]} did not happen: one of you was in two swaps at once.")
            continue
        a, b = agents[ia], agents[ib]
        a["pos"], b["pos"] = b["pos"], a["pos"]
        moved |= {ia, ib}
        bank.cooldown[ia] = bank.cooldown[ib] = t + int(rc["move_cooldown"]) + 1
        events.append({"kind": "swap", "a": ia, "b": ib, "from": addr[ia], "to": addr[ib]})
        news[ia].append(f"swap done: your address is now {addr[ib]}.")
        news[ib].append(f"swap done: your address is now {addr[ia]}.")
    if moved:
        _rewire_meta(meta)
    return moved


def _take_proposals(meta, bank, t, raws, texts, moved, results) -> list[dict]:
    agents = meta["agents"]
    at = {address(a): a["id"] for a in agents}
    out = []
    for i, text in enumerate(texts):
        if raws[i] is None:
            continue
        found = _MOVE.findall(text)
        if not found:
            continue
        note = results[i]["notices"]
        if len(found) > 1:
            note.append("only your first <move> was read.")
        target = at.get(found[0].lower())
        if i in moved:
            note.append("<move> was ignored: you swapped this tick and your address changed.")
        elif t < bank.cooldown[i]:
            note.append(f"<move> was ignored: you cannot swap again until tick {bank.cooldown[i]}.")
        elif target is None or target not in neighbours(agents[i]):
            note.append(f"<move> was ignored: {found[0].lower()} is not next to you.")
        elif target in moved:
            note.append(f"<move> was ignored: the process at {found[0].lower()} changed this tick.")
        elif t < bank.cooldown[target]:
            note.append(f"<move> was ignored: {found[0].lower()} cannot swap until tick {bank.cooldown[target]}.")
        else:
            out.append({"a": i, "b": target, "t": t})
    return out


def interesting(window: list[dict], before: list[dict], n: int) -> tuple[bool, dict]:
    """ratchet signal: a value was delivered in the window, a program survived its gate, or at least 5% of nodes changed
    their public register and at least 3 distinct non-zero values appeared on it."""
    start = before[-1]["reg"] if before and before[-1].get("reg") else None
    prev = start["pub"] if start else None
    changed, seen = set(), set()
    delivered = 0
    for rec in window:
        reg = rec.get("reg") or {}
        pubs = reg.get("pub") or []
        delivered = reg.get("delivered", delivered)
        for i, v in enumerate(pubs):
            if prev is not None and v != prev[i]:
                changed.add(i)
            if v.strip("0"):
                seen.add(v)
        prev = pubs or prev
    gained = delivered - (start["delivered"] if start else 0)
    kept = sum(e["kind"] == "program_kept" for rec in window for e in (rec.get("reg") or {}).get("events", []))
    keep = gained > 0 or kept > 0 or (len(changed) >= 0.05 * n and len(seen) >= 3)
    return keep, {"delivered_gain": gained, "programs_kept": kept, "public_changers": len(changed), "public_values": len(seen),
                  "registers": keep}


# ---------------------------------------------------------------- a fake agent that solves the task

class FakeRegisterAgent:
    """tests the plumbing without a model, and speaks the real reply format. needs tell_cells. the input node puts the
    value's bytes on the bus in turn, high on odd ticks, low on even ticks; the output node rebuilds it. per-pool noise
    garbles what a node publishes. it proposes swaps with its right-hand neighbour at move_rate and accepts any swap."""

    def __init__(self, cfg: dict):
        self.noise = dict(cfg.get("noise") or {})
        self.move_rate = float(cfg.get("move_rate") or 0)
        self.malformed_rate = float(cfg.get("malformed_rate") or 0)
        self.script = {str(k): v for k, v in (cfg.get("script") or {}).items()}   # "agent:tick" -> extra reply text

    def gluon(self, turn, rng):
        """follows its instruction exactly, except that pool noise garbles what it writes."""
        from sim import Reply
        st, p = turn.state, turn.prompt
        have = {f"p{k}": v for k, v in enumerate(re.search(r"^p0 (\w+)  p1 (\w+)  p2 (\w+)  p3 (\w+)$", p, re.M).groups())}
        addr, pub, priv = re.search(r"your address (\w\w)  public (\w+)  private (\w+)", p).groups()
        have.update(public=pub, private=priv, **{"private.hi": priv[:len(priv) // 2], "private.lo": priv[len(priv) // 2:]})
        m = re.search(r"^input (\w+)$", p, re.M)
        if m:
            v = m.group(1)
            have.update(input=v, **{"input.hi": v[:len(v) // 2], "input.lo": v[len(v) // 2:]})
        out = {f"q{k}": v for k, v in enumerate(st.outputs)}
        out.update(public=pub, private=priv)
        written = None
        for src, dst in re.findall(r"copy (\S+) to ([^\s;]+)", p.split("\n", 1)[0]):
            if src not in have:
                continue
            v = have[src]
            if rng.random() < float(self.noise.get(st.pool, 0)):
                v = "".join("f" if c != "f" else "0" for c in v)
            if dst == "out":
                written = v
            elif dst == "private.hi":
                out["private"] = v + out["private"][len(v):]
            elif dst == "private.lo":
                out["private"] = out["private"][:-len(v)] + v
            else:
                out[dst] = v
        extra = f"<out>{written}</out>\n" if written and "output register here" in p else ""
        body = "".join(f"<q{i}>{out[f'q{i}']}</q{i}>\n" for i in range(4))
        return Reply(raw=f"<reg>{addr} {out['public']} {out['private']}</reg>\n<log>ok</log>\n{extra}{body}")

    async def act(self, turn):
        from sim import Reply   # imported here: sim imports this module at startup
        st, t, p = turn.state, turn.t, turn.prompt
        rng = random.Random(f"{st.seed}:{t}")
        if rng.random() < self.malformed_rate:
            return Reply(raw="hmm. i am not sure what any of this means.")
        if turn.system.startswith("you pass values along"):
            return self.gluon(turn, rng)
        addr = re.search(r"your address: (\w\w)", p).group(1)
        pub = re.search(r"your public register: (\w+)", p).group(1)
        priv = re.search(r"your private register: (\w+)", p).group(1)
        board = dict(re.findall(r"\b([0-9a-f]{2}):([0-9a-f]{2})\b", p))
        told = re.search(r"input register is at position (\w\w)", turn.system)
        input_at = told.group(1) if told else ""
        extra = self.script.get(f"{st.id}:{t}", "")
        extra = extra + "\n" if extra else ""
        m = re.search(r"input register at your position: (\w+)", p)
        if m:
            pub = m.group(1)[:2] if t % 2 else m.group(1)[2:4]
            if rng.random() < float(self.noise.get(st.pool, 0)):
                pub = f"{(int(pub, 16) ^ 0xf0):02x}"
        if "output register at your position" in p:
            seen = board.get(input_at, "00")
            if t % 2:              # read on an odd tick: the input node published the low byte on the even tick before
                extra += f"<out>{priv[:2]}{seen}</out>\n"
            else:
                priv = seen + priv[2:]
        for asker in re.findall(r"(\w\w) asks to swap with you", p):
            extra += f'<yes from="{asker}"/>\n'
        if rng.random() < self.move_rate:
            cols = 1 + max(int(a[1], 16) for a in board)
            extra += f'<move to="{addr[0]}{(int(addr[1], 16) + 1) % cols:x}"/>\n'
        body = "".join(f"<q{i}>{v}</q{i}>\n" for i, v in enumerate(st.outputs))
        return Reply(raw=f"<reg>{addr} {pub} {priv}</reg>\n<log>fake</log>\n{extra}{body}")

    async def close(self):
        pass


# ---------------------------------------------------------------- summary

def summary(path: Path) -> dict:
    meta, ticks = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta = rec
            elif rec.get("type") == "tick":
                ticks.append(rec)
    if not meta or not on(meta["config"]):
        sys.exit(f"{path}: not a register run")
    rc = meta["config"]["registers"]
    period, last = max(1, int(rc["input_period"])), ticks[-1] if ticks else None
    delivered = [e for r in ticks for e in r["reg"]["events"] if e["kind"] == "delivered"]
    swaps = [e for r in ticks for e in r["reg"]["events"] if e["kind"] == "swap"]
    count = lambda kind: sum(e["kind"] == kind for r in ticks for e in r["reg"]["events"])
    refused = [e for r in ticks for e in r["reg"]["events"] if e["kind"] == "swap_refused"]
    pools = {i: a.get("pool", "") for i, a in enumerate(meta["agents"])}
    stats: dict[str, dict] = {}
    for r in ticks:
        for a in r["agents"]:
            s = stats.setdefault(pools[a["agent"]] or "all", {"replies": 0, "led_ok": 0, "led_first": 0})
            s["replies"] += 1
            s["led_ok"] += bool(a.get("reg_ok"))
            s["led_first"] += bool(a.get("led_first"))
    lat = sorted(e["latency"] for e in delivered)
    where = {}
    if last:
        for name in ("input_cell", "output_cell"):
            cell = str(rc[name]).lower()
            i = last["reg"]["addr"].index(cell)
            where[name] = {"address": cell, "node": i, "pool": pools[i]}
    return {
        "run_id": meta["run_id"],
        "ticks": len(ticks),
        "values_shown": (ticks[-1]["t"] - 1) // period + 1 if ticks else 0,
        "values_delivered": len(delivered),
        "mean_signal_last_quarter": round(sum(r["reg"]["signal"] for r in ticks[-max(1, len(ticks) // 4):]) / max(1, len(ticks[-max(1, len(ticks) // 4):])), 4),
        "chance_per_write": f"1 in {16 ** int(rc['input_hex'])}",
        "median_latency": lat[len(lat) // 2] if lat else None,
        "swaps": len(swaps),
        "swaps_refused": len(refused),
        "programs": {k: count(f"program_{k}") for k in ("trial", "kept", "failed_test", "reverted", "refused")},
        "led_line": stats,
        "at_the_end": where,
    }


if __name__ == "__main__":
    print(json.dumps(summary(Path(sys.argv[1])), indent=2))
