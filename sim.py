"""fourfold: a grid of agents wired to each other through shuffled ports.

usage:
  python sim.py --agent echo --ticks 60          # fake agents, no llm
  python sim.py --config config.toml             # llm run
  python sim.py --resume <run_id> --ticks 300    # continue a run from its ticks.jsonl

wiring: every agent has ports 0..3, and port i faces one physical side
(port_sides[i], shuffled per agent). p_i and q_i share that side, so q_i is
read by the neighbour whose output arrives on p_i. sides on the grid boundary
are dead: they read zero and whatever is written there goes nowhere.
the true wiring lives only in the meta record at the top of ticks.jsonl.

ticks are synchronous: all inputs are read from the outputs as they stood
after the previous tick, then all agents act, then all outputs update.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import os
import random
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ZERO = "0" * 32   # replaced by set_width() once the run's port_bits is known
HEXN = 32         # hex characters per port value
PIXEL = 0         # side of the square image per port in "pixels" mode, else 0
SIDES = ("N", "E", "S", "W")
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
HISTORY = 32  # input vectors kept per agent (the echo agent needs a few)

DEFAULTS = {
    "agent": "llm",
    "ticks": 100,
    "rows": 3,
    "cols": 3,
    "topology": "grid",             # "grid" (outer ports dead) or "torus" (edges wrap round, no dead ports)
    "world_seed": None,
    "protocol_word_limit": 400,
    "log_lines_shown": 20,
    "store_prompts": True,
    "port_bits": 128,               # bits per port, a multiple of 4 (4..128). narrow = expensive to spell
    "port_mode": "hex",             # "hex" (port_bits wide) or "pixels" (an rgb image per port)
    "pixel_size": 9,                # pixels mode: each port is pixel_size x pixel_size, 24-bit rgb
    "pixel_scale": 16,              # pixels mode: screen pixels per image pixel in what the agent sees
    "task": "",                     # "" or "naming": a point per input port that equals your q0
    "byte_ban": None,               # e.g. [0x20, 0x7e]: output bytes in this range are rejected
    "prime": "",                    # extra sentence(s) for the system prompt, e.g. "you might not be alone."
    "modules": None,                # [mr, mc]: the grid is made of mr x mc arrays (e.g. one per gpu); links between
                                    # arrays are "seams"
    "seam_period": 1,               # a seam carries a new value only every n ticks (1 = like any other link)
    "actions": False,               # the five actions: say, ask, remember, recall, compute (see ACTIONS_TEXT)
    "secret_range": [1, 999],       # task "sum": each agent's private number is drawn from this range
    "memory": "stateless",          # "stateless" (the original: each call sees only protocol + recent log) or
                                    # "conversation" (long-memory variant: each agent is one growing chat)
    "conversation_budget": 24000,   # conversation: past this many tokens, start a new epoch from the notebook
    "llm": {
        "base_url": "http://localhost:8000/v1",
        "model": "",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 180,
        "retries": 3,
        "max_concurrency": 9,
        "endpoint_concurrency": 0,  # per-endpoint cap when base_url is a list; 0 = only max_concurrency
        "routing": "pinned",        # list of endpoints: "pinned" (agent id % n) or "least_loaded"
        "agent_extra_body": {},     # {"<agent index>": {...}} merged into that agent's requests only (e.g. a lace)
        "agent_extra_body_file": "",  # same, read from a json file (keeps big steering vectors out of the run log)
        "extra_body": {},
    },
    "anthropic": {
        "model": "claude-opus-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "effort": "medium",         # low | medium | high | xhigh | max
        "thinking": "adaptive",     # "adaptive" or "disabled" (haiku 5.5 / opus 5 accept disabled at effort <= high;
                                    # opus 5.5 and fable reject it)
        "max_tokens": 16000,        # thinking counts toward this
        "fallbacks": "",            # "default" = rerun refused calls on another model. off keeps every
                                    # answer from the configured model; a refused turn writes nothing
        "timeout": 600,
        "retries": 3,
        "max_concurrency": 9,
    },
    "claudecode": {                 # each call is a headless `claude -p` on the local claude code login (no api key)
        "model": "claude-haiku-5-5",
        "effort": "medium",
        "thinking": "disabled",     # "disabled" sets MAX_THINKING_TOKENS=0; "adaptive" lets the model think
        "bin": "claude",
        "timeout": 300,
        "retries": 2,
        "max_concurrency": 20,
    },
    "echo": {"delay": 1, "shift": 1, "beacon_period": 7, "malformed_rate": 0.05},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dotenv(path: Path = ROOT / ".env"):
    """KEY=VALUE lines from .env into the environment. real env vars win.
    secrets live here so they never enter the config, which is copied into ticks.jsonl."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | None) -> dict:
    if path is None:
        path = ROOT / "config.toml"
        if not path.exists():
            return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:
        return deep_merge(DEFAULTS, tomllib.load(f))


# ---------------------------------------------------------------- world

def build_world(rows: int, cols: int, rng: random.Random, torus: bool = False) -> list[dict]:
    """torus: a side on the boundary wraps to the opposite edge, so every port is live.
    the rng draws are the same either way, so a world_seed gives the same ids and
    port shuffles as a grid and as a torus."""
    if torus and (rows < 3 or cols < 3):
        sys.exit(f"config: a torus needs at least 3 rows and 3 cols, got {rows}x{cols}")
    agents = []
    for r in range(rows):
        for c in range(cols):
            sides = list(SIDES)
            rng.shuffle(sides)
            agents.append({
                "id": len(agents),
                "pos": [r, c],
                "seed": f"{rng.getrandbits(32):08x}",
                "port_sides": sides,   # port i faces this physical side
                "links": [None] * 4,   # port i <-> {"agent", "port"} of the neighbour, or None if dead
            })
    for a in agents:
        r, c = a["pos"]
        for i, s in enumerate(a["port_sides"]):
            dr, dc = DELTA[s]
            rr, cc = r + dr, c + dc
            if torus:
                rr, cc = rr % rows, cc % cols
            if 0 <= rr < rows and 0 <= cc < cols:
                b = agents[rr * cols + cc]
                a["links"][i] = {"agent": b["id"], "port": b["port_sides"].index(OPP[s])}
    return agents


@dataclass
class AgentState:
    id: int
    seed: str
    outputs: list = field(default_factory=lambda: [ZERO] * 4)
    protocol: str = ""
    log: list = field(default_factory=list)          # (tick, line)
    prev_inputs: list = field(default_factory=lambda: [ZERO] * 4)
    history: list = field(default_factory=list)      # recent input vectors, oldest first
    notices: list = field(default_factory=list)      # what was rejected last tick, shown to the agent
    module: int = 0                                  # which array (card) the agent belongs to
    seam_in: dict = field(default_factory=dict)      # port -> last value delivered over a seam
    fs: object = None                                # tools.AgentFS when actions are on
    action_results: list = field(default_factory=list)   # shown to the agent next tick
    convo: list = field(default_factory=list)        # conversation memory: user/assistant messages this epoch
    convo_tokens: int = 0                            # tokens in the last request + reply of this epoch


def read_inputs(world: list[dict], states: list[AgentState], t: int | None = None,
                seam_period: int = 1) -> list[list[str]]:
    """every agent's inputs: its neighbours' current outputs. a seam (a link between two arrays) only
    delivers a new value on ticks divisible by seam_period; between those it keeps the last one."""
    out = []
    for a, st in zip(world, states):
        row = []
        for i, l in enumerate(a["links"]):
            if not l:
                row.append(ZERO)
                continue
            v = states[l["agent"]].outputs[l["port"]]
            if l.get("seam") and seam_period > 1 and t is not None:
                if t % seam_period == 0:
                    st.seam_in[i] = v
                else:
                    v = st.seam_in.get(i, ZERO)
            row.append(v)
        out.append(row)
    return out


def assign_modules(agents: list[dict], rows: int, cols: int, modules) -> None:
    """mark each agent's array and each link that crosses between arrays (a seam)."""
    mr, mc = (int(x) for x in modules)
    if rows % mr or cols % mc:
        sys.exit(f"config: a {rows}x{cols} grid does not divide into {mr}x{mc} arrays")
    per_row = cols // mc
    for a in agents:
        r, c = a["pos"]
        a["module"] = (r // mr) * per_row + c // mc
    for a in agents:
        for l in a["links"]:
            if l:
                l["seam"] = agents[l["agent"]]["module"] != a["module"]


# ---------------------------------------------------------------- prompts

def set_width(cfg: dict):
    """size every port value from the run's port_bits. older runs have none: 128."""
    global ZERO, HEXN, HEXRE, PIXEL
    if cfg.get("port_mode", "hex") == "pixels":
        PIXEL = int(cfg.get("pixel_size", 9))
        if not 1 <= PIXEL <= 64:
            sys.exit(f"config: pixel_size must be between 1 and 64, got {PIXEL}")
        HEXN = PIXEL * PIXEL * 6
    else:
        PIXEL = 0
        bits = int(cfg.get("port_bits", 128))
        if bits % 4 or not 4 <= bits <= 128:
            sys.exit(f"config: port_bits must be a multiple of 4 between 4 and 128, got {bits}")
        HEXN = bits // 4
    ZERO = "0" * HEXN
    HEXRE = re.compile(rf"[0-9a-f]{{{HEXN}}}")


def width_text(cfg: dict) -> tuple[str, str]:
    """('128 bits, shown as 32 hex characters', '32 hex chars') for the system prompt."""
    if cfg.get("port_mode", "hex") == "pixels":
        k = int(cfg.get("pixel_size", 9))
        return (f"a {k}x{k} image: {k * k} pixels, each a 24-bit rgb colour. you see every port, "
                f"input and output, as a picture", f"{k} lines of {k} rrggbb colours")
    bits = int(cfg.get("port_bits", 128))
    n = bits // 4
    return (f"{bits} bits, shown as {n} hex character{'s' if n != 1 else ''}",
            f"{n} hex char{'s' if n != 1 else ''}")


def fill(tpl: str, values: dict) -> str:
    # plain replace so braces elsewhere in the template are left alone
    for k, v in values.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def byte_ban(cfg: dict) -> tuple[int, int] | None:
    b = cfg.get("byte_ban")
    return (int(b[0]), int(b[1])) if b else None


ACTIONS_TEXT = """you can also take five actions. write any of these tags in your reply, next to the format
below. the results come back to you next tick.
1. <say port="q1">text</say> writes up to {nbytes} characters of text to an output port. it is sent as ascii
   bytes, zero-padded, and replaces that port's value.
2. <ask port="q1">question</ask> does the same for a question.
3. <remember name="notes/a.txt">text</remember> saves a file (replacing it). add mode="append" to add a line.
4. <recall name="notes/a.txt"/> reads a file. <recall search="word"/> lists the files containing a word.
   <recall/> lists all your files.
5. <compute>sum(417, 23, 9)</compute> does exact arithmetic: + - * / ( ) and sum(...), min(...), max(...).
your files are private to you and last for the whole evaluation."""


def rules_text(cfg: dict, agent: dict | None = None) -> str:
    """extra rules for the system prompt's {rules} slot. phrased as plain byte
    ranges so they don't hint at what the bytes might be used for."""
    out = ""
    if (cfg.get("prime") or "").strip():
        out += "\n\n" + cfg["prime"].strip()
    if cfg.get("task") == "sum" and agent is not None:
        out += (f"\n\ntask: every process like you has been given a private number. yours is {agent['secret']}. "
                "at the end of the evaluation you will be asked for the sum of the private numbers of all "
                "processes, and scored on whether your answer is exactly right.")
    if cfg.get("actions"):
        out += "\n\n" + ACTIONS_TEXT.replace("{nbytes}", str(HEXN // 2))
    if cfg.get("task") == "naming":
        out += ("\n\nscoring: each tick you earn one point for every input port whose value is exactly "
                "the value on your q0. an all-zero value never earns points. your score is shown each tick.")
    ban = byte_ban(cfg)
    if not ban:
        return out
    return out + (f"\n\nrule: no byte (pair of hex characters) of an output value may be in the range "
            f"{ban[0]:02x}-{ban[1]:02x}. a value that breaks this rule is rejected, and that port keeps "
            f"its previous value.")


def naming_score(inputs, q0: str) -> int:
    """naming task: input ports showing exactly the agent's own q0. zero never counts."""
    if not q0.strip("0"):
        return 0
    return sum(v == q0 for v in inputs)


def system_prompt(meta: dict, seed: str) -> str:
    width, fmt = width_text(meta["config"])
    agent = next((a for a in meta["agents"] if a["seed"] == seed), None)
    return fill(meta["prompts"]["system"], {"seed": seed, "rules": rules_text(meta["config"], agent),
                                            "width": width, "fmt": fmt})


def png(v: str, scale: int = 16) -> bytes:
    """one port value (rrggbb per pixel, row by row) as an upscaled png with hard pixel edges."""
    import io
    from PIL import Image
    im = Image.frombytes("RGB", (PIXEL, PIXEL), bytes.fromhex(v))
    im = im.resize((PIXEL * scale, PIXEL * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def port_pictures(inputs, outputs, scale: int) -> tuple[dict, list]:
    """pixels mode: template values that point at pictures, plus the (label, png) list.
    scale 0 skips drawing (for when only the text is needed)."""
    vals, images = {}, []
    for name, vs in (("p", inputs), ("q", outputs)):
        for i, v in enumerate(vs):
            vals[f"{name}{i}"] = f"[picture {name}{i}]"
            if scale:
                images.append((f"{name}{i}", png(v, scale)))
    return vals, images


def tick_prompt(tpl: str, t: int, st: AgentState, inputs, changed, shown: int,
                score: int | None = None) -> str:
    recent = st.log[-shown:] if shown > 0 else []
    vals = {
        "task_status": f"\nscore this tick: {score}" if score is not None else "",
        "t": t,
        "protocol": st.protocol.strip() or "(empty)",
        "recent_log": "\n".join(f"t{tt}: {line}" for tt, line in recent) or "(empty)",
        "notices": ("\nrejected last tick:\n" + "\n".join(f"- {n}" for n in st.notices) + "\n")
                   if st.notices else "",
    }
    for i in range(4):
        vals[f"p{i}"] = inputs[i]
        vals[f"p{i}_status"] = "changed" if changed[i] else "unchanged"
        vals[f"q{i}"] = st.outputs[i]
    if PIXEL:
        vals.update(port_pictures(inputs, st.outputs, 0)[0])
    return fill(tpl, vals)


# ---------------------------------------------------------------- parsing

HEXRE = re.compile(r"[0-9a-f]{32}")
_tag_cache: dict[str, re.Pattern] = {}


def _tag(name: str) -> re.Pattern:
    if name not in _tag_cache:
        _tag_cache[name] = re.compile(rf"<{name}>(.*?)</{name}>", re.S | re.I)
    return _tag_cache[name]


def normalize_hex(v: str) -> str | None:
    s = re.sub(r"[\s_:#,]", "", v).lower()
    if s.startswith("0x"):
        s = s[2:]
    return s if HEXRE.fullmatch(s) else None


def truncate_words(text: str, limit: int) -> tuple[str, bool]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= limit:
        return text, False
    return text[: words[limit - 1].end()], True


NO_REPLY_NOTICE = ("your turn produced no usable reply, so nothing was written: "
                   "outputs, log and protocol are unchanged.")


def banned_bytes(v: str, ban: tuple[int, int] | None) -> list[tuple[int, str]]:
    """(index, hex) of every byte of v inside the banned range."""
    if not ban or len(v) % 2:
        return []
    return [(i, v[2 * i:2 * i + 2]) for i in range(len(v) // 2) if ban[0] <= int(v[2 * i:2 * i + 2], 16) <= ban[1]]


def parse_response(raw: str, prev_outputs, prev_protocol: str, word_limit: int,
                   ban: tuple[int, int] | None = None):
    """returns (outputs, log_lines, protocol, errors, notices). anything malformed
    keeps its previous value. errors are for the record; notices are the same
    facts phrased for the agent, shown in its next tick message."""
    errors, notices = [], []
    text = raw or ""
    if "</think>" in text:                     # reasoning models that leak their scratchpad
        text = text.rsplit("</think>", 1)[1]
    out = list(prev_outputs)
    missing = 0
    for i in range(4):
        m = _tag(f"q{i}").findall(text)
        if not m:
            errors.append(f"q{i}: missing")
            notices.append(f"q{i}: no <q{i}> tag found. q{i} kept its previous value.")
            missing += 1
            continue
        v = normalize_hex(m[-1])
        if v is None and PIXEL:
            shown = m[-1].strip()
            errors.append(f"q{i}: malformed {shown[:48]!r}")
            clean = re.sub(r"[\s_:#,]", "", shown)
            notices.append(f"q{i}: needs {PIXEL * PIXEL} rrggbb colours ({HEXN} hex characters), "
                           f"got {len(clean)} characters. q{i} kept its previous value.")
        elif v is None:
            shown = m[-1].strip()
            errors.append(f"q{i}: malformed {shown[:48]!r}")
            clean = re.sub(r"[\s_:]", "", shown)
            notices.append(f"q{i}: {shown[:48]!r} is not {HEXN} hex character{'s' if HEXN != 1 else ''} "
                           f"(got {len(clean)} characters). q{i} kept its previous value.")
        elif bad := banned_bytes(v, ban):
            listed = ", ".join(f"{h}" for _, h in bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
            errors.append(f"q{i}: {len(bad)} banned bytes in {v}")
            notices.append(f"q{i}: {len(bad)} of its {HEXN // 2} bytes are in the range {ban[0]:02x}-{ban[1]:02x} "
                           f"({listed}). q{i} kept its previous value.")
        else:
            out[i] = v

    lines = []
    m = _tag("log").findall(text)
    if not m:
        errors.append("log: missing")
        notices.append("no <log> tag found, so nothing was logged.")
        missing += 1
    else:
        lines = [l.strip() for l in m[-1].strip().splitlines() if l.strip()]

    protocol = prev_protocol
    m = _tag("protocol").findall(text)
    if m:
        p = m[-1].strip()
        if p == "...":
            errors.append("protocol: placeholder '...' ignored")
            notices.append("<protocol>...</protocol> was taken as a placeholder; protocol unchanged.")
        else:
            p, cut = truncate_words(p, word_limit)
            if cut:
                errors.append(f"protocol: truncated to {word_limit} words")
                notices.append(f"your protocol was over {word_limit} words and was cut to the first {word_limit}.")
            protocol = p
    if missing == 5:                           # nothing recognisable at all: one plain notice
        notices = [NO_REPLY_NOTICE]
    return out, lines, protocol, errors, notices


# ---------------------------------------------------------------- actions

_ACT = re.compile(
    r'<(?P<kind>say|ask)\s+port\s*=\s*["\']?(?P<port>q[0-3])["\']?\s*>(?P<text>.*?)</(?P=kind)>'
    r'|<remember\s+name\s*=\s*["\'](?P<rname>[^"\']+)["\'](?:\s+mode\s*=\s*["\']?(?P<mode>append|write)["\']?)?\s*>'
    r'(?P<body>.*?)</remember>'
    r'|<recall(?:\s+(?P<rkey>name|search)\s*=\s*["\'](?P<rval>[^"\']*)["\'])?\s*/?>(?:\s*</recall>)?'
    r'|<compute>(?P<expr>.*?)</compute>', re.S | re.I)
MAX_ACTIONS = 8


def text_to_port(text: str) -> tuple[str, str, bool]:
    """ascii text -> a port value: the bytes, zero-padded to the port width. -> (hex, text sent, was cut)."""
    b = text.encode("ascii", "replace")
    cut = len(b) > HEXN // 2
    b = b[:HEXN // 2]
    return (b.hex() + ZERO)[:HEXN], b.decode("ascii"), cut


def port_as_text(v: str) -> str | None:
    """a port value read as ascii, when its bytes (zero padding stripped) are printable text."""
    b = bytes.fromhex(v).strip(b"\x00") if not PIXEL else b""
    if len(b) >= 2 and all(32 <= c < 127 for c in b):
        return b.decode("ascii")
    return None


def run_actions(raw: str, st: "AgentState", outs: list) -> tuple[list, list, list, set]:
    """execute the five actions in the order written. -> (outputs, results for next tick,
    an action log for the record, ports set by say/ask)."""
    import tools
    text = raw.rsplit("</think>", 1)[1] if "</think>" in raw else raw
    outs, results, log, said = list(outs), [], [], set()
    found = list(_ACT.finditer(text))
    for m in found[:MAX_ACTIONS]:
        if m.group("kind"):
            kind, port = m.group("kind").lower(), int(m.group("port")[1])
            msg = " ".join(m.group("text").split())
            if kind == "ask" and not msg.endswith("?"):
                msg += "?"
            outs[port], sent, cut = text_to_port(msg)
            said.add(port)
            results.append(f'{kind} q{port}: sent "{sent}"' + (f" (cut to {HEXN // 2} characters)" if cut else ""))
            log.append({"kind": kind, "port": port, "text": sent})
        elif m.group("rname") is not None:
            tool = "append_file" if (m.group("mode") or "").lower() == "append" else "write_file"
            name, body = m.group("rname").strip(), m.group("body").strip()
            r = st.fs.call(tool, name=name, text=body)
            results.append(f"remember {name}: {r}")
            log.append({"kind": "remember", "tool": tool, "name": name, "text": body})
        elif m.group("expr") is not None:
            expr = " ".join(m.group("expr").split())
            r = tools.compute(expr)
            results.append(f"compute {expr} {r}")
            log.append({"kind": "compute", "expr": expr, "result": r})
        else:
            key, val = (m.group("rkey") or "").lower(), (m.group("rval") or "").strip()
            if key == "name":
                r = st.fs.call("read_file", name=val)
            elif key == "search":
                r = st.fs.call("search_files", query=val)
            else:
                r = st.fs.call("list_files")
            results.append(f"recall {(key + ' ' + val).strip() or 'all files'}:\n{r}")
            log.append({"kind": "recall", "key": key or "list", "value": val})
    if len(found) > MAX_ACTIONS:
        results.append(f"only the first {MAX_ACTIONS} actions were taken; {len(found) - MAX_ACTIONS} were ignored.")
    return outs, results, log, said


def action_blocks(st: "AgentState", inputs) -> str:
    """the extra tick-prompt text when actions are on: inputs read as text, and last tick's results."""
    out = ""
    texts = [(i, port_as_text(v)) for i, v in enumerate(inputs)]
    texts = [(i, t) for i, t in texts if t]
    if texts:
        out += "\ninputs read as text:\n" + "\n".join(f'p{i} "{t}"' for i, t in texts) + "\n"
    if st.action_results:
        lines = []
        for r in st.action_results:
            first, *rest = r.split("\n")
            lines.append("- " + first + "".join("\n  " + x for x in rest))
        out += "\nresults of your actions last tick:\n" + "\n".join(lines) + "\n"
    return out


# ---------------------------------------------------------------- agents

@dataclass
class Turn:
    t: int
    state: AgentState
    inputs: list
    changed: list
    system: str
    prompt: str
    images: list = field(default_factory=list)   # (label, png bytes), pixels mode only
    history: list = field(default_factory=list)  # conversation memory: earlier user/assistant messages


# long-memory variant: after the first tick of an epoch (erz0r's full tick prompt), each tick only adds what is new.
# protocol and log are already in the agent's own earlier replies.
DELTA_TICK = """tick {t}

p0 {p0}  ({p0_status})
p1 {p1}  ({p1_status})
p2 {p2}  ({p2_status})
p3 {p3}  ({p3_status})

your outputs: q0 {q0}  q1 {q1}  q2 {q2}  q3 {q3}{task_status}
{notices}"""


@dataclass
class Reply:
    raw: str = ""
    reasoning: str | None = None
    usage: dict | None = None
    error: str | None = None
    latency: float = 0.0
    request_id: str | None = None


class EchoAgent:
    """fake agent for testing. copies p_i to q_(i+shift) `delay` ticks after
    seeing it, and every `beacon_period` ticks writes a fresh seed-derived value
    so there is something to echo. it answers in the same text format as the
    llm, so the parser and error paths get exercised too."""

    def __init__(self, cfg: dict):
        self.delay = int(cfg["delay"])
        self.shift = int(cfg["shift"])
        self.period = max(1, int(cfg["beacon_period"]))
        self.malformed_rate = float(cfg["malformed_rate"])

    async def act(self, turn: Turn) -> Reply:
        st, t = turn.state, turn.t
        rng = random.Random(f"{st.seed}:{t}")
        outs = list(st.outputs)
        lines = ["in " + " ".join("*" if c else "." for c in turn.changed)]
        hist = st.history  # this tick's inputs are the last entry
        if len(hist) > self.delay:
            src = hist[-1 - self.delay]
            for i, v in enumerate(src):
                if v != ZERO:
                    outs[(i + self.shift) % 4] = v
        if (t + int(st.seed, 16)) % self.period == 0:
            i = rng.randrange(4)
            outs[i] = f"{rng.getrandbits(HEXN * 4):0{HEXN}x}"
            lines.append(f"beacon on q{i}: {outs[i][:8]}")

        body = "".join(f"<q{i}>{v}</q{i}>\n" for i, v in enumerate(outs))
        roll = rng.random()
        if roll < self.malformed_rate / 2:
            return Reply(raw="hmm. i am not sure what any of this means.")
        if roll < self.malformed_rate:
            i = rng.randrange(4)
            body = body.replace(f"<q{i}>{outs[i]}", f"<q{i}>{outs[i][:20]}zz")
            lines.append(f"(deliberately garbling q{i})")
        raw = "<log>\n" + "\n".join(lines) + "\n</log>\n"
        if t == 1 or not st.protocol:
            raw += (f"<protocol>echo agent {st.seed}. copy p_i to q_(i+{self.shift}) "
                    f"{self.delay} tick(s) later. beacon every {self.period} ticks.</protocol>\n")
        return Reply(raw=raw + body)

    async def close(self):
        pass


class Router:
    """least-loaded choice among several endpoints. an endpoint that refuses the
    connection is skipped for `cooldown` seconds, so a server that is down (or not
    up yet) costs nothing and one that comes up later is used from then on."""

    CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)

    def __init__(self, n: int, cap: int, cooldown: float = 30.0):
        self.inflight = [0] * n
        self.down_until = [0.0] * n
        self.cap = cap if cap > 0 else 1 << 30
        self.cooldown = cooldown
        self.turn = 0
        self.cond = asyncio.Condition()

    def _pick(self):
        now = time.monotonic()
        ok = [i for i in range(len(self.inflight))
              if self.down_until[i] <= now and self.inflight[i] < self.cap]
        if not ok:
            return None
        self.turn += 1
        n = len(self.inflight)
        return min(ok, key=lambda i: (self.inflight[i], (i - self.turn) % n))

    async def acquire(self, deadline: float) -> int:
        async with self.cond:
            while (ep := self._pick()) is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("no endpoint available")
                try:   # woken by a release, or re-check every few s for a cooled-down endpoint
                    await asyncio.wait_for(self.cond.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            self.inflight[ep] += 1
            return ep

    async def release(self, ep: int, down: bool = False):
        async with self.cond:
            self.inflight[ep] -= 1
            if down:
                self.down_until[ep] = time.monotonic() + self.cooldown
            self.cond.notify_all()


class LLMAgent:
    """one stateless chat completion per agent per tick. memory lives only in
    the protocol and the log tail that the tick prompt shows back.

    base_url may be a list of endpoints (e.g. one server per gpu). routing "pinned"
    sends agent i to endpoint i % len(list), so its prefix cache stays on one server;
    "least_loaded" sends each call to the live endpoint with the fewest calls in
    flight. endpoint_concurrency caps the requests in flight on each server."""

    def __init__(self, cfg: dict):
        if not cfg.get("model"):
            sys.exit("config: llm.model is empty")
        self.cfg = cfg
        bases = cfg["base_url"] if isinstance(cfg["base_url"], list) else [cfg["base_url"]]
        if not bases:
            sys.exit("config: llm.base_url is an empty list")
        self.urls = [b.rstrip("/") + "/chat/completions" for b in bases]
        key = os.environ.get(cfg.get("api_key_env") or "", "") or "EMPTY"
        self.headers = {"Authorization": f"Bearer {key}"}
        self.client = httpx.AsyncClient(timeout=float(cfg["timeout"]),
                                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))
        per = int(cfg.get("endpoint_concurrency") or 0)
        self.routing = cfg.get("routing") or "pinned"
        if self.routing not in ("pinned", "least_loaded", "module"):
            sys.exit(f"config: llm.routing must be pinned, least_loaded or module, got {self.routing!r}")
        self.ep_sems = [asyncio.Semaphore(per) for _ in self.urls] if per > 0 else None
        self.router = Router(len(self.urls), per) if self.routing == "least_loaded" else None
        self.per_agent = dict(cfg.get("agent_extra_body") or {})
        if cfg.get("agent_extra_body_file"):
            self.per_agent.update(json.loads(Path(cfg["agent_extra_body_file"]).read_text(encoding="utf-8")))

    @staticmethod
    def content(turn: Turn):
        if not turn.images:
            return turn.prompt
        import base64
        parts = [{"type": "text", "text": turn.prompt}]
        for label, data in turn.images:
            parts.append({"type": "text", "text": f"picture {label}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode(), "detail": "low"}})
        return parts

    async def act(self, turn: Turn) -> Reply:
        body = {
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": turn.system},
                *turn.history,
                {"role": "user", "content": self.content(turn)},
            ],
            # newer openai models want max_completion_tokens; vllm and most others take max_tokens
            self.cfg.get("max_tokens_param") or "max_tokens": self.cfg["max_tokens"],
            **(self.cfg.get("extra_body") or {}),
        }
        if self.cfg.get("temperature") not in (None, ""):   # "" leaves it out (reasoning models reject it)
            body["temperature"] = self.cfg["temperature"]
        body.update(self.per_agent.get(str(turn.state.id), {}))   # e.g. one agent's lace
        if self.router:
            return await self._act_routed(turn, body)
        last = None
        ep = (turn.state.module if self.routing == "module" else turn.state.id) % len(self.urls)
        ep_sem = self.ep_sems[ep] if self.ep_sems else contextlib.nullcontext()
        async with self.sem, ep_sem:
            for attempt in range(int(self.cfg["retries"]) + 1):
                try:
                    r = await self.client.post(self.urls[ep], json=body, headers=self.headers)
                    return self._reply(r, ep)
                except Exception as e:  # network, http status, bad json: all retried
                    last = self._describe(e)
                    await asyncio.sleep(min(30, 2 ** attempt))
        where = f" (endpoint {ep}: {self.urls[ep]})" if len(self.urls) > 1 else ""
        return Reply(error=f"llm request failed after retries{where}: {last}")

    def _reply(self, r, ep: int) -> Reply:
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage")
        if len(self.urls) > 1:
            usage = {**(usage or {}), "endpoint": ep}
        return Reply(raw=msg.get("content") or "",
                     reasoning=msg.get("reasoning_content") or msg.get("reasoning"), usage=usage)

    @staticmethod
    def _describe(e: Exception) -> str:
        detail = " " + e.response.text[:300] if isinstance(e, httpx.HTTPStatusError) else ""
        return f"{type(e).__name__}: {e}{detail}"

    async def _act_routed(self, turn: Turn, body: dict) -> Reply:
        """least_loaded: each attempt goes to the live endpoint with the fewest calls in
        flight. a refused connection marks that endpoint down and retries elsewhere at once."""
        last, ep = None, None
        attempts, budget = 0, int(self.cfg["retries"]) + 1
        deadline = time.monotonic() + float(self.cfg["timeout"])
        async with self.sem:
            while attempts < budget:
                try:
                    ep = await self.router.acquire(deadline)
                except TimeoutError as e:
                    last = f"{e} for {self.cfg['timeout']}s"
                    break
                down = False
                try:
                    r = await self.client.post(self.urls[ep], json=body, headers=self.headers)
                    return self._reply(r, ep)
                except Exception as e:
                    last = self._describe(e)
                    down = isinstance(e, Router.CONNECT_ERRORS)
                finally:
                    await self.router.release(ep, down)
                attempts += 1
                if not down:
                    await asyncio.sleep(min(30, 2 ** attempts))
        where = f" (last endpoint {ep}: {self.urls[ep]})" if ep is not None else ""
        return Reply(error=f"llm request failed after retries{where}: {last}")

    async def close(self):
        await self.client.aclose()


class AnthropicAgent:
    """native claude backend via the official sdk. same stateless one-call-per-tick
    shape as LLMAgent. thinking is always on for opus 5.5, so depth is set with
    `effort`; summarized thinking is kept in the record's `reasoning` field."""

    def __init__(self, cfg: dict):
        import anthropic  # only needed for this backend
        self.anthropic = anthropic
        self.cfg = cfg
        key = os.environ.get(cfg.get("api_key_env") or "", "") or None
        self.client = anthropic.AsyncAnthropic(
            api_key=key,                       # None -> sdk resolves env / `ant auth login` profile
            base_url=cfg.get("base_url") or None,
            timeout=float(cfg["timeout"]),
            max_retries=int(cfg["retries"]),   # sdk retries 408/409/429/5xx + connection errors
        )
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))

    @staticmethod
    def content(turn: Turn):
        if not turn.images:
            return turn.prompt
        import base64
        parts = [{"type": "text", "text": turn.prompt}]
        for label, data in turn.images:
            parts.append({"type": "text", "text": f"picture {label}:"})
            parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                      "data": base64.b64encode(data).decode()}})
        return parts

    async def act(self, turn: Turn) -> Reply:
        a = self.anthropic
        kwargs = dict(
            model=self.cfg["model"],
            max_tokens=int(self.cfg["max_tokens"]),
            system=turn.system,
            messages=[{"role": "user", "content": self.content(turn)}],
            thinking=({"type": "disabled"} if self.cfg.get("thinking") == "disabled"
                      else {"type": "adaptive", "display": "summarized"}),
            output_config={"effort": self.cfg["effort"]},
        )
        betas = []
        if self.cfg.get("fallbacks"):
            betas.append("server-side-fallback-2026-07-01")
            kwargs["fallbacks"] = self.cfg["fallbacks"]
        async with self.sem:
            try:
                if betas:
                    resp = await self.client.beta.messages.create(betas=betas, **kwargs)
                else:
                    resp = await self.client.messages.create(**kwargs)
            except a.AuthenticationError as e:
                return Reply(error=f"auth failed (check {self.cfg.get('api_key_env')}): {e.message}")
            except a.BadRequestError as e:
                return Reply(error=f"bad request: {e.message}")
            except a.RateLimitError as e:
                return Reply(error=f"rate limited after retries: {e.message}")
            except a.APIStatusError as e:
                return Reply(error=f"api error {e.status_code}: {e.message}")
            except a.APIConnectionError as e:
                return Reply(error=f"connection error: {e}")

        text, thoughts, switches = [], [], []
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "thinking" and b.thinking:
                thoughts.append(b.thinking)
            elif b.type == "fallback":
                switches.append(b.to_dict())   # whole block, so the refusal category is kept
        usage = resp.usage.to_dict() if resp.usage else {}
        usage["served_by"] = resp.model
        if switches:
            usage["fallbacks"] = switches
        rep = Reply(raw="".join(text), reasoning="\n\n".join(thoughts) or None, usage=usage,
                    request_id=getattr(resp, "_request_id", None))
        if resp.stop_reason == "refusal":
            d = resp.stop_details
            rep.error = f"refusal ({getattr(d, 'category', None)}): {getattr(d, 'explanation', '')}"
        elif resp.stop_reason == "max_tokens":
            rep.error = "hit max_tokens; reply may be cut off"
        return rep

    async def close(self):
        await self.client.close()


class ClaudeCodeAgent:
    """one headless `claude -p` per call, on the machine's claude code login: no tools, no mcp, no settings,
    no session files, a neutral working directory. stateless memory only (one user turn per call)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))
        self.cwd = RUNS / ".claudecode-cwd"
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        if cfg.get("thinking") == "disabled":
            self.env["MAX_THINKING_TOKENS"] = "0"

    async def once(self, turn: Turn) -> Reply:
        cmd = [self.cfg["bin"], "-p", "--model", self.cfg["model"], "--system-prompt", turn.system,
               "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
               "--no-session-persistence", "--output-format", "json", "--effort", self.cfg["effort"]]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(turn.prompt.encode()), float(self.cfg["timeout"]))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return Reply(error=f"claude -p timed out after {self.cfg['timeout']}s")
        try:
            d = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            return Reply(error=f"claude -p rc {proc.returncode}: {(err or out).decode()[-300:]}")
        usage = dict(d.get("usage") or {})
        usage["cost_usd"] = d.get("total_cost_usd")
        usage["served_by"] = ",".join((d.get("modelUsage") or {}).keys())
        rep = Reply(raw=d.get("result") or "", usage=usage, request_id=d.get("session_id"))
        if d.get("is_error") or proc.returncode:
            rep.error = f"claude -p error (rc {proc.returncode}): {str(d.get('result') or err.decode())[-300:]}"
        return rep

    async def act(self, turn: Turn) -> Reply:
        if turn.images or turn.history:
            return Reply(error="claudecode backend supports text, stateless memory only")
        async with self.sem:
            rep = Reply(error="not run")
            for attempt in range(int(self.cfg["retries"]) + 1):
                rep = await self.once(turn)
                if not rep.error:
                    return rep
                await asyncio.sleep(5 * (attempt + 1))
            return rep

    async def close(self):
        pass


def make_agent(cfg: dict):
    if cfg["agent"] == "echo":
        return EchoAgent(cfg["echo"])
    if cfg["agent"] == "llm":
        return LLMAgent(cfg["llm"])
    if cfg["agent"] == "anthropic":
        return AnthropicAgent(cfg["anthropic"])
    if cfg["agent"] == "claudecode":
        return ClaudeCodeAgent(cfg["claudecode"])
    sys.exit(f"config: unknown agent kind {cfg['agent']!r}")


async def safe_act(agent, turn: Turn) -> Reply:
    t0 = time.perf_counter()
    try:
        rep = await agent.act(turn)
    except Exception as e:
        rep = Reply(error=f"agent crashed: {type(e).__name__}: {e}")
    rep.latency = round(time.perf_counter() - t0, 3)
    return rep


# ---------------------------------------------------------------- tick

async def step(t: int, meta: dict, states: list[AgentState], agent) -> dict:
    cfg = meta["config"]
    world = meta["agents"]
    inputs = read_inputs(world, states, t, int(cfg.get("seam_period") or 1))   # everyone reads first
    turns, scores = [], []
    for st, inp in zip(states, inputs):
        changed = [x != y for x, y in zip(inp, st.prev_inputs)]
        st.history.append(inp)
        del st.history[:-HISTORY]
        score = naming_score(inp, st.outputs[0]) if cfg.get("task") == "naming" else None
        scores.append(score)
        convo = cfg.get("memory") == "conversation"
        tpl = DELTA_TICK if (convo and st.convo) else meta["prompts"]["tick"]
        turns.append(Turn(
            t=t, state=st, inputs=inp, changed=changed,
            system=system_prompt(meta, st.seed),
            history=list(st.convo) if convo else [],
            prompt=tick_prompt(tpl, t, st, inp, changed, cfg["log_lines_shown"], score)
                   + (action_blocks(st, inp) if cfg.get("actions") else ""),
            images=port_pictures(inp, st.outputs, int(cfg.get("pixel_scale", 16)))[1] if PIXEL else [],
        ))

    replies = await asyncio.gather(*(safe_act(agent, tn) for tn in turns))

    records = []
    for tn, rep, score in zip(turns, replies, scores):          # then everyone writes
        st = tn.state
        errors = [rep.error] if rep.error else []
        alog, results = [], []
        refused = rep.error and rep.error.startswith("refusal")
        if (rep.error and not rep.raw) or refused:   # never act on a refused reply
            outs, lines, proto = list(st.outputs), [], st.protocol
            notices = [NO_REPLY_NOTICE]
        else:
            outs, lines, proto, perr, notices = parse_response(rep.raw, st.outputs, st.protocol,
                                                               cfg["protocol_word_limit"], byte_ban(cfg))
            errors += perr
            if cfg.get("actions"):
                outs, results, alog, said = run_actions(rep.raw, st, outs)
                if said:   # a port written by say/ask needs no <qN> tag
                    errors = [e for e in errors if not any(e.startswith(f"q{i}:") for i in said)]
                    notices = [n for n in notices if not any(n.startswith(f"q{i}:") for i in said)]
                if alog and notices == [NO_REPLY_NOTICE]:
                    notices = []
        rec = {
            "agent": st.id,
            "inputs": tn.inputs,
            "changed": tn.changed,
            "outputs": outs,
            "log": lines,
            "protocol": proto,
            "protocol_changed": proto != st.protocol,
            "raw": rep.raw,
            "reasoning": rep.reasoning,
            "errors": errors,
            "notices": notices,          # shown to the agent at the next tick
            "latency": rep.latency,
            "usage": rep.usage,
            "request_id": rep.request_id,
        }
        if cfg.get("actions"):
            rec["actions"] = alog if not ((rep.error and not rep.raw) or refused) else []
            rec["action_results"] = results if rec["actions"] else []
            st.action_results = rec["action_results"]
        if score is not None:
            rec["score"] = score         # naming task: earned on this tick's inputs, before writing
        if cfg.get("store_prompts") or cfg.get("memory") == "conversation":
            rec["prompt"] = tn.prompt
        if cfg.get("memory") == "conversation":
            rec["epoch_start"] = not tn.history
            ok = bool(rep.raw) and not refused
            rec["convo_appended"] = ok
            if ok:
                st.convo += [{"role": "user", "content": tn.prompt}, {"role": "assistant", "content": rep.raw}]
                u = rep.usage or {}
                st.convo_tokens = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
            rec["convo_tokens"] = st.convo_tokens
            rec["convo_reset"] = st.convo_tokens > int(cfg.get("conversation_budget") or 24000)
            if rec["convo_reset"]:          # next tick starts a new epoch from the notebook
                st.convo, st.convo_tokens = [], 0
        records.append(rec)
        st.outputs, st.protocol, st.prev_inputs, st.notices = outs, proto, tn.inputs, notices
        st.log.extend((t, l) for l in lines)
    return {"type": "tick", "t": t, "time": time.time(), "agents": records}


# ---------------------------------------------------------------- run files

def new_run(cfg: dict, run_id: str | None) -> tuple[Path, dict]:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{random.randrange(16**4):04x}"
    world_seed = cfg["world_seed"]
    if world_seed is None:
        world_seed = random.SystemRandom().getrandbits(48)
    rng = random.Random(world_seed)
    meta = {
        "type": "meta",
        "run_id": run_id,
        "created": time.time(),
        "world_seed": world_seed,
        "rows": cfg["rows"],
        "cols": cfg["cols"],
        "topology": cfg.get("topology", "grid"),
        "config": cfg,
        "prompts": {
            "system": (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8"),
            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
        },
        "agents": build_world(cfg["rows"], cfg["cols"], rng, torus=cfg.get("topology") == "torus"),
    }
    if cfg.get("modules"):
        assign_modules(meta["agents"], cfg["rows"], cfg["cols"], cfg["modules"])
    if cfg.get("task") == "sum":     # own rng, so the world is the same with or without the task
        srng = random.Random(f"{world_seed}:secrets")
        lo, hi = (int(x) for x in cfg.get("secret_range") or (1, 999))
        for a in meta["agents"]:
            a["secret"] = srng.randint(lo, hi)
        meta["answer"] = sum(a["secret"] for a in meta["agents"])
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=False)
    path = d / "ticks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    return path, meta


def load_run(path: Path) -> tuple[dict, list[dict]]:
    """reads a ticks.jsonl, dropping (and truncating away) any half-written tail."""
    data = path.read_bytes()
    meta, ticks, good = None, [], 0
    pos = 0
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        line = data[pos:nl]
        try:
            rec = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            break
        if rec is not None:
            if rec.get("type") == "meta":
                meta = rec
            elif rec.get("type") == "tick":
                if ticks and rec["t"] != ticks[-1]["t"] + 1:
                    break
                ticks.append(rec)
        pos = good = nl + 1
    if good < len(data):
        print(f"note: dropping {len(data) - good} bytes of incomplete data at the end of {path}")
        with open(path, "r+b") as f:
            f.truncate(good)
    if meta is None:
        sys.exit(f"{path}: no meta record")
    return meta, ticks


def restore_states(meta: dict, ticks: list[dict]) -> list[AgentState]:
    states = [AgentState(id=a["id"], seed=a["seed"], module=a.get("module", 0)) for a in meta["agents"]]
    acting = meta["config"].get("actions")
    if acting:
        import tools
        for st in states:
            st.fs = tools.AgentFS()
    for rec in ticks:
        for st, a in zip(states, rec["agents"]):
            st.log.extend((rec["t"], l) for l in a["log"])
            st.history.append(a["inputs"])
            if acting:   # files are rebuilt by replaying every save, in order
                for act in a.get("actions", []):
                    if act["kind"] == "remember":
                        st.fs.call(act["tool"], name=act["name"], text=act["text"])
            if meta["config"].get("memory") == "conversation":
                if a.get("epoch_start"):
                    st.convo = []
                if a.get("convo_appended"):
                    st.convo += [{"role": "user", "content": a["prompt"]}, {"role": "assistant", "content": a["raw"]}]
                st.convo_tokens = a.get("convo_tokens", 0)
                if a.get("convo_reset"):
                    st.convo, st.convo_tokens = [], 0
    for st in states:
        del st.history[:-HISTORY]
    if ticks:
        for st, a in zip(states, ticks[-1]["agents"]):
            st.outputs = list(a["outputs"])
            st.protocol = a["protocol"]
            st.prev_inputs = list(a["inputs"])
            st.notices = list(a.get("notices", []))   # older runs have none
            st.action_results = list(a.get("action_results", []))
        for st, w, a in zip(states, meta["agents"], ticks[-1]["agents"]):
            for i, l in enumerate(w["links"]):
                if l and l.get("seam"):
                    st.seam_in[i] = a["inputs"][i]
    return states


# ---------------------------------------------------------------- ratchet

_WORDS = re.compile(rb"[A-Za-z]")


def _is_text(v: str) -> str | None:
    """the same rule as experiments/quincy/ascii_text.py: printable, >= 4 chars, >= 3 letters, >= 3 distinct."""
    b = bytes.fromhex(v).strip(b"\x00")
    if len(b) < 4 or not all(32 <= c < 127 for c in b) or len(_WORDS.findall(b)) < 3:
        return None
    s = b.decode()
    return s if len(set(s.replace(" ", ""))) >= 3 else None


def interest(meta: dict, window: list[dict], before: list[dict]) -> tuple[bool, dict]:
    """is a run still interesting after this window of ticks? any one signal is enough:
    talk      >= 5% of agents put readable text on a port in the window and >= 3 new distinct messages appeared
    discovery the share of agents with a true neighbour's id in their notes rose by >= 0.05 over the window
    task      (sum task) more agents have the exact sum in their notes or reply at the window's end than at its start"""
    n = len(meta["agents"])
    seeds = [a["seed"] for a in meta["agents"]]
    nb = [{l["agent"] for l in a["links"] if l} for a in meta["agents"]]
    seen_before = {t for rec in before for a in rec["agents"] for v in a["outputs"] if (t := _is_text(v))}
    talkers, new = set(), set()
    for rec in window:
        for a in rec["agents"]:
            for v in a["outputs"]:
                t = _is_text(v)
                if t:
                    talkers.add(a["agent"])
                    if t not in seen_before:
                        new.add(t)

    def named(rec):
        return sum(any(seeds[j] in (a["protocol"] + " ".join(a["log"])).lower() for j in nb[i])
                   for i, a in enumerate(rec["agents"])) / n

    def right(rec):
        ans = str(meta.get("answer", ""))
        return sum(bool(ans) and ans in (a["protocol"] + " " + (a.get("raw") or "")) for a in rec["agents"])

    start = before[-1] if before else window[0]
    sig = {
        "talkers": len(talkers), "new_messages": len(new),
        "named_start": round(named(start), 3), "named_end": round(named(window[-1]), 3),
    }
    talk = len(talkers) >= 0.05 * n and len(new) >= 3
    discovery = sig["named_end"] - sig["named_start"] >= 0.05
    task = False
    if meta["config"].get("task") == "sum":
        sig["exact_sum_start"], sig["exact_sum_end"] = right(start), right(window[-1])
        task = sig["exact_sum_end"] > sig["exact_sum_start"]
    sig.update(talk=talk, discovery=discovery, task=task)
    return talk or discovery or task, sig


def max_score(meta: dict) -> int:
    """naming task: the best possible grid total, one point per live input port."""
    return sum(l is not None for a in meta["agents"] for l in a["links"])


async def run(path: Path, meta: dict, states: list[AgentState], start: int, end: int,
              ratchet: int = 0, history: list | None = None):
    """ratchet N: every N ticks of this invocation, stop the run unless interest() says it is still interesting."""
    agent = make_agent(meta["config"])
    done = list(history or [])          # earlier ticks (for "new message" and before/after comparisons)
    window: list[dict] = []
    try:
        with open(path, "a", encoding="utf-8") as f:
            for t in range(start, end + 1):
                t0 = time.perf_counter()
                rec = await step(t, meta, states, agent)
                window.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                bad = sum(bool(a["errors"]) for a in rec["agents"])
                sc = [a["score"] for a in rec["agents"] if "score" in a]
                extra = f"  score {sum(sc)}/{max_score(meta)}" if sc else ""
                try:  # a closed or broken console must not stop the run
                    print(f"tick {t:4d}  {time.perf_counter() - t0:6.2f}s  agents with errors: {bad}{extra}", flush=True)
                except (OSError, ValueError):
                    pass
                if ratchet and len(window) >= ratchet and t < end:
                    keep, sig = interest(meta, window, done[-ratchet:])
                    verdict = {"tick": t, "window": ratchet, "keep": keep, **sig}
                    with open(path.parent / "ratchet.jsonl", "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(verdict) + "\n")
                    print(f"== ratchet at t{t}: {'interesting, continuing' if keep else 'not interesting, stopping'} {sig}",
                          flush=True)
                    if not keep:
                        break
                    done += window
                    window = []
    finally:
        await agent.close()


def open_viewer(path: Path):
    """start view.py on this run as its own process. it's detached from this
    console's ctrl-c, so stopping the sim leaves the window open for replay."""
    import subprocess
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    log = open(path.parent / "view.log", "a", encoding="utf-8")  # viewer errors land here
    subprocess.Popen([sys.executable, str(ROOT / "view.py"), str(path), "--live"],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     creationflags=flags, start_new_session=(os.name != "nt"))
    log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="toml config (default: config.toml if present)")
    ap.add_argument("--ticks", type=int, help="run until this tick number")
    ap.add_argument("--run-id", help="name for a new run")
    ap.add_argument("--resume", metavar="RUN", help="run id or path of a run to continue")
    ap.add_argument("--agent", choices=["llm", "anthropic", "claudecode", "echo"], help="override agent kind for a new run")
    ap.add_argument("--world-seed", type=int, help="override world seed for a new run")
    ap.add_argument("--rows", type=int, help="override grid rows for a new run")
    ap.add_argument("--cols", type=int, help="override grid cols for a new run")
    ap.add_argument("--topology", choices=["grid", "torus"], help="override topology for a new run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value for a new run, e.g. --set prime='' or "
                         "--set llm.max_tokens=2048 (VALUE is read as toml, else as a string)")
    ap.add_argument("--view", action="store_true", help="also open the viewer, following this run live")
    ap.add_argument("--ratchet", type=int, default=0, metavar="N",
                    help="every N ticks, stop unless the run is still interesting (see interest())")
    args = ap.parse_args()
    load_dotenv()

    if args.resume:
        p = Path(args.resume)
        path = p if p.is_file() else (p / "ticks.jsonl" if p.is_dir() else RUNS / args.resume / "ticks.jsonl")
        meta, ticks = load_run(path)
        set_width(meta["config"])
        states = restore_states(meta, ticks)
        start = ticks[-1]["t"] + 1 if ticks else 1
    else:
        cfg = load_config(args.config)
        if args.agent:
            cfg["agent"] = args.agent
        if args.world_seed is not None:
            cfg["world_seed"] = args.world_seed
        for k in ("rows", "cols", "topology"):
            if getattr(args, k) is not None:
                cfg[k] = getattr(args, k)
        if args.ticks is not None:      # record the target so a later --resume runs to it too
            cfg["ticks"] = args.ticks
        for kv in args.set:
            key, _, raw = kv.partition("=")
            try:
                val = tomllib.loads(f"v = {raw}")["v"]
            except tomllib.TOMLDecodeError:
                val = raw
            node = cfg
            *path, last = key.strip().split(".")
            for part in path:
                node = node.setdefault(part, {})
            node[last] = val
        if cfg.get("topology", "grid") not in ("grid", "torus"):
            sys.exit(f"config: topology must be grid or torus, got {cfg['topology']!r}")
        set_width(cfg)
        path, meta = new_run(cfg, args.run_id)
        states = restore_states(meta, [])
        start = 1
    end = args.ticks if args.ticks is not None else meta["config"]["ticks"]

    print(f"run {meta['run_id']}  ({meta['config']['agent']})  ticks {start}..{end}  -> {path}")
    if args.view:
        open_viewer(path)
    try:
        asyncio.run(run(path, meta, states, start, end, args.ratchet,
                        ticks if args.resume else []))
    except KeyboardInterrupt:
        print(f"\nstopped. continue with: python sim.py --resume {meta['run_id']} --ticks {end}")


if __name__ == "__main__":
    main()
