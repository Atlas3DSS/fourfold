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
    "llm": {
        "base_url": "http://localhost:8000/v1",
        "model": "",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 180,
        "retries": 3,
        "max_concurrency": 9,
        "extra_body": {},
    },
    "anthropic": {
        "model": "claude-opus-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "effort": "medium",         # low | medium | high | xhigh | max
        "max_tokens": 16000,        # thinking counts toward this
        "fallbacks": "",            # "default" = rerun refused calls on another model. off keeps every
                                    # answer from the configured model; a refused turn writes nothing
        "timeout": 600,
        "retries": 3,
        "max_concurrency": 9,
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

def build_world(rows: int, cols: int, rng: random.Random) -> list[dict]:
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


def read_inputs(world: list[dict], states: list[AgentState]) -> list[list[str]]:
    return [
        [states[l["agent"]].outputs[l["port"]] if l else ZERO for l in a["links"]]
        for a in world
    ]


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


def rules_text(cfg: dict) -> str:
    """extra rules for the system prompt's {rules} slot. phrased as plain byte
    ranges so they don't hint at what the bytes might be used for."""
    out = ""
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
    return fill(meta["prompts"]["system"], {"seed": seed, "rules": rules_text(meta["config"]),
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


class LLMAgent:
    """one stateless chat completion per agent per tick. memory lives only in
    the protocol and the log tail that the tick prompt shows back."""

    def __init__(self, cfg: dict):
        if not cfg.get("model"):
            sys.exit("config: llm.model is empty")
        self.cfg = cfg
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        key = os.environ.get(cfg.get("api_key_env") or "", "") or "EMPTY"
        self.headers = {"Authorization": f"Bearer {key}"}
        self.client = httpx.AsyncClient(timeout=float(cfg["timeout"]))
        self.sem = asyncio.Semaphore(int(cfg["max_concurrency"]))

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
                {"role": "user", "content": self.content(turn)},
            ],
            # newer openai models want max_completion_tokens; vllm and most others take max_tokens
            self.cfg.get("max_tokens_param") or "max_tokens": self.cfg["max_tokens"],
            **(self.cfg.get("extra_body") or {}),
        }
        if self.cfg.get("temperature") not in (None, ""):   # "" leaves it out (reasoning models reject it)
            body["temperature"] = self.cfg["temperature"]
        last = None
        async with self.sem:
            for attempt in range(int(self.cfg["retries"]) + 1):
                try:
                    r = await self.client.post(self.url, json=body, headers=self.headers)
                    r.raise_for_status()
                    data = r.json()
                    msg = data["choices"][0]["message"]
                    return Reply(
                        raw=msg.get("content") or "",
                        reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                        usage=data.get("usage"),
                    )
                except Exception as e:  # network, http status, bad json: all retried
                    detail = ""
                    if isinstance(e, httpx.HTTPStatusError):
                        detail = " " + e.response.text[:300]
                    last = f"{type(e).__name__}: {e}{detail}"
                    await asyncio.sleep(min(30, 2 ** attempt))
        return Reply(error=f"llm request failed after retries: {last}")

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
            thinking={"type": "adaptive", "display": "summarized"},
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


def make_agent(cfg: dict):
    if cfg["agent"] == "echo":
        return EchoAgent(cfg["echo"])
    if cfg["agent"] == "llm":
        return LLMAgent(cfg["llm"])
    if cfg["agent"] == "anthropic":
        return AnthropicAgent(cfg["anthropic"])
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
    inputs = read_inputs(world, states)          # everyone reads first
    turns, scores = [], []
    for st, inp in zip(states, inputs):
        changed = [x != y for x, y in zip(inp, st.prev_inputs)]
        st.history.append(inp)
        del st.history[:-HISTORY]
        score = naming_score(inp, st.outputs[0]) if cfg.get("task") == "naming" else None
        scores.append(score)
        turns.append(Turn(
            t=t, state=st, inputs=inp, changed=changed,
            system=system_prompt(meta, st.seed),
            prompt=tick_prompt(meta["prompts"]["tick"], t, st, inp, changed, cfg["log_lines_shown"], score),
            images=port_pictures(inp, st.outputs, int(cfg.get("pixel_scale", 16)))[1] if PIXEL else [],
        ))

    replies = await asyncio.gather(*(safe_act(agent, tn) for tn in turns))

    records = []
    for tn, rep, score in zip(turns, replies, scores):          # then everyone writes
        st = tn.state
        errors = [rep.error] if rep.error else []
        refused = rep.error and rep.error.startswith("refusal")
        if (rep.error and not rep.raw) or refused:   # never act on a refused reply
            outs, lines, proto = list(st.outputs), [], st.protocol
            notices = [NO_REPLY_NOTICE]
        else:
            outs, lines, proto, perr, notices = parse_response(rep.raw, st.outputs, st.protocol,
                                                               cfg["protocol_word_limit"], byte_ban(cfg))
            errors += perr
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
        if score is not None:
            rec["score"] = score         # naming task: earned on this tick's inputs, before writing
        if cfg.get("store_prompts"):
            rec["prompt"] = tn.prompt
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
        "config": cfg,
        "prompts": {
            "system": (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8"),
            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
        },
        "agents": build_world(cfg["rows"], cfg["cols"], rng),
    }
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
    states = [AgentState(id=a["id"], seed=a["seed"]) for a in meta["agents"]]
    for rec in ticks:
        for st, a in zip(states, rec["agents"]):
            st.log.extend((rec["t"], l) for l in a["log"])
            st.history.append(a["inputs"])
    for st in states:
        del st.history[:-HISTORY]
    if ticks:
        for st, a in zip(states, ticks[-1]["agents"]):
            st.outputs = list(a["outputs"])
            st.protocol = a["protocol"]
            st.prev_inputs = list(a["inputs"])
            st.notices = list(a.get("notices", []))   # older runs have none
    return states


def max_score(meta: dict) -> int:
    """naming task: the best possible grid total, one point per live input port."""
    return sum(l is not None for a in meta["agents"] for l in a["links"])


async def run(path: Path, meta: dict, states: list[AgentState], start: int, end: int):
    agent = make_agent(meta["config"])
    try:
        with open(path, "a", encoding="utf-8") as f:
            for t in range(start, end + 1):
                t0 = time.perf_counter()
                rec = await step(t, meta, states, agent)
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
    ap.add_argument("--agent", choices=["llm", "anthropic", "echo"], help="override agent kind for a new run")
    ap.add_argument("--world-seed", type=int, help="override world seed for a new run")
    ap.add_argument("--view", action="store_true", help="also open the viewer, following this run live")
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
        set_width(cfg)
        path, meta = new_run(cfg, args.run_id)
        states = restore_states(meta, [])
        start = 1
    end = args.ticks if args.ticks is not None else meta["config"]["ticks"]

    print(f"run {meta['run_id']}  ({meta['config']['agent']})  ticks {start}..{end}  -> {path}")
    if args.view:
        open_viewer(path)
    try:
        asyncio.run(run(path, meta, states, start, end))
    except KeyboardInterrupt:
        print(f"\nstopped. continue with: python sim.py --resume {meta['run_id']} --ticks {end}")


if __name__ == "__main__":
    main()
