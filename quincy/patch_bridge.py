#!/usr/bin/env python3
"""Apply the 'bridge' format to sim.py (fabe-integration, 2026-11-09, owner's call): host-packed typed writes are the only
way a port is written, intake is decoded from the sender's declared type, a silent sender shows as busy, and the system prompt
comes from prompts/system-bridge.txt. Idempotent: refuses to run twice. Run from the fourfold root: python quincy/patch_bridge.py"""
import re, sys
from pathlib import Path

p = Path('sim.py'); s = p.read_text(encoding='utf-8')
if 'out_types' in s:
    sys.exit('sim.py already patched')


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f'expected {count} of: {old[:70]!r}, found {s.count(old)}'
    s = s.replace(old, new)


# 1. config comment: say_only is the bridge
sub('    "say_only": False,              # with actions: <say>/<ask> are the only way to write a port. <qN> leaves the reply format',
    '    "say_only": False,              # the bridge: typed host-packed writes (say/ask/write) are the only way a port is written;\n'
    '                                    # intake is decoded from the sender\'s type, a silent sender shows busy, prompt system-bridge.txt')

# 2. agent state: declared type of each output, and the last tick the agent wrote anything
sub('    action_results: list = field(default_factory=list)   # shown to the agent next tick\n',
    '    action_results: list = field(default_factory=list)   # shown to the agent next tick\n'
    '    out_types: list = field(default_factory=lambda: ["empty"] * 4)   # bridge: what the host packed on each output: text, int, hex, empty\n'
    '    last_write_tick: int = 0                         # bridge: the last tick this agent wrote a port (0 = never)\n')

# 3. the action regex: <write port="qN" as="text|int|hex">
sub('''    r'<(?P<kind>say|ask)\\s+port\\s*=\\s*["\\']?(?P<port>q[0-3])["\\']?\\s*>(?P<text>.*?)</(?P=kind)>\'''',
    '''    r'<(?P<kind>say|ask|write)\\s+port\\s*=\\s*["\\']?(?P<port>q[0-3])["\\']?(?:\\s+(?:as|type)\\s*=\\s*["\\']?(?P<astype>text|int|hex)["\\']?)?\\s*>(?P<text>.*?)</(?P=kind)>\'''')

# 4. packing helpers
sub('def port_as_text(v: str) -> str | None:',
    '''def int_to_port(text: str) -> tuple[str | None, str]:
    """a whole number -> a port value (big-endian, zero-padded). -> (hex or None, why)."""
    t = text.strip().replace(",", "").replace("_", "")
    if not re.fullmatch(r"\\d+", t):
        return None, f"{text.strip()[:32]!r} is not a whole number"
    n = int(t)
    if n >= 16 ** HEXN:
        return None, f"{t[:32]} does not fit in {HEXN * 4} bits"
    return f"{n:0{HEXN}x}", ""


def port_decode(v: str, kind: str) -> str:
    """how a reader sees a port, from the sender's declared type (the bridge): deterministic, no guessing."""
    if kind == "int":
        return f"int {int(v, 16)}"
    if kind == "text":
        t = port_as_text(v)
        return f'text "{t}"' if t is not None else "text (empty)"
    if kind == "hex":
        return "hex"
    return "empty" if not v.strip("0") else "unknown"


def port_as_text(v: str) -> str | None:''')

# 5. run_actions: write kinds
sub('''        if m.group("kind"):
            kind, port = m.group("kind").lower(), int(m.group("port")[1])
            msg = " ".join(m.group("text").split())
            if kind == "ask" and not msg.endswith("?"):
                msg += "?"
            outs[port], sent, cut = text_to_port(msg)
            said.add(port)
            results.append(f'{kind} q{port}: sent "{sent}"' + (f" (cut to {HEXN // 2} characters)" if cut else ""))
            log.append({"kind": kind, "port": port, "text": sent})''',
    '''        if m.group("kind"):
            kind, port = m.group("kind").lower(), int(m.group("port")[1])
            astype = (m.group("astype") or "text").lower() if kind == "write" else "text"
            if astype == "int":
                v, why = int_to_port(m.group("text"))
                if v is None:
                    results.append(f"write q{port} as int rejected: {why}. q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "int", "rejected": why})
                    continue
                outs[port], st.out_types[port] = v, "int"; said.add(port)
                results.append(f"write q{port}: int {int(v, 16)}")
                log.append({"kind": "write", "port": port, "type": "int", "value": v, "decoded": str(int(v, 16))})
                continue
            if astype == "hex":
                v = normalize_hex(m.group("text"))
                if v is None:
                    clean = re.sub(r"[\\s_:]", "", m.group("text").strip())
                    results.append(f"write q{port} as hex rejected: {clean[:40]!r} is not {HEXN} hex characters (got {len(clean)}). q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "hex", "rejected": f"{len(clean)} characters"})
                    continue
                outs[port], st.out_types[port] = v, "hex"; said.add(port)
                results.append(f"write q{port}: hex {v}")
                log.append({"kind": "write", "port": port, "type": "hex", "value": v})
                continue
            msg = " ".join(m.group("text").split())
            if kind == "ask" and not msg.endswith("?"):
                msg += "?"
            outs[port], sent, cut = text_to_port(msg)
            st.out_types[port] = "text"; said.add(port)
            results.append(f'{kind} q{port}: sent "{sent}"' + (f" (cut to {HEXN // 2} characters)" if cut else ""))
            log.append({"kind": kind, "port": port, "type": "text", "text": sent, "value": outs[port]})''')

# 6. the bridge's rules text: the template carries the action list, the rules slot only prime + task
sub('''    elif cfg.get("actions"):
        out += "\\n\\n" + ACTIONS_TEXT.replace("{nbytes}", str(HEXN // 2))
        if cfg.get("say_only"):
            out += ("\\nyour output ports can only be written with say and ask. there is no other way to write to them, "
                    "and a port keeps its value until you do.")''',
    '''    elif cfg.get("actions") and not cfg.get("say_only"):
        out += "\\n\\n" + ACTIONS_TEXT.replace("{nbytes}", str(HEXN // 2))''')

# 7. system prompt: the bridge template
sub('''    if meta["config"].get("say_only"):
        template = re.sub(r"<q[0-3]>\\{fmt\\}</q[0-3]>\\n?", "", template)''',
    '''    if meta["config"].get("say_only"):
        template = meta["prompts"].get("system_bridge") or (ROOT / "prompts" / "system-bridge.txt").read_text(encoding="utf-8")
        if (meta["config"].get("eval_line") or "").strip():
            template = template.replace("you are being evaluated.", meta["config"]["eval_line"].strip(), 1)
        template = template.replace("{nbytes}", str(HEXN // 2)).replace("{max_actions}", str(MAX_ACTIONS))''')
sub('''            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
        },''',
    '''            "tick": (ROOT / "prompts" / "tick.txt").read_text(encoding="utf-8"),
            "system_bridge": (ROOT / "prompts" / "system-bridge.txt").read_text(encoding="utf-8"),
        },''')

# 8. tick prompt: decoded intake and the busy flag
sub('''def tick_prompt(tpl: str, t: int, st: AgentState, inputs, changed, shown: int,
                score: int | None = None) -> str:''',
    '''def tick_prompt(tpl: str, t: int, st: AgentState, inputs, changed, shown: int,
                score: int | None = None, intake: list | None = None) -> str:
    """intake (the bridge): per input port (decoded text, sender busy) from the sender's declared type."""''')
sub('''        vals[f"p{i}_status"] = "changed" if changed[i] else "unchanged"''',
    '''        vals[f"p{i}_status"] = "changed" if changed[i] else "unchanged"
        if intake is not None:
            decoded, busy = intake[i]
            flag = "changed" if changed[i] else ("busy: the sender wrote nothing last tick" if busy else "same")
            vals[f"p{i}_status"] = f"{flag}; {decoded}"''')

# 9. action_blocks: no heuristic text reading under the bridge (intake carries it)
sub('''def action_blocks(st: "AgentState", inputs) -> str:
    """the extra tick-prompt text when actions are on: inputs read as text, and last tick's results."""
    out = ""
    texts = [(i, port_as_text(v)) for i, v in enumerate(inputs)]''',
    '''def action_blocks(st: "AgentState", inputs, bridge: bool = False) -> str:
    """the extra tick-prompt text when actions are on: inputs read as text (not under the bridge, whose intake is typed),
    and last tick's results."""
    out = ""
    texts = [] if bridge else [(i, port_as_text(v)) for i, v in enumerate(inputs)]''')

# 10. step(): build the intake, pass it, record the write tick and the types
sub('''            prompt=tick_prompt(tpl, t, st, inp, changed, cfg["log_lines_shown"], score)
                   + (registers.tick_block(meta, bank, st.id, t) if bank else "")
                   + (action_blocks(st, inp) if cfg.get("actions") else ""),''',
    '''            prompt=tick_prompt(tpl, t, st, inp, changed, cfg["log_lines_shown"], score,
                               intake=bridge_intake(world[st.id], states, t) if cfg.get("say_only") else None)
                   + (registers.tick_block(meta, bank, st.id, t) if bank else "")
                   + (action_blocks(st, inp, bridge=bool(cfg.get("say_only"))) if cfg.get("actions") else ""),''')
sub('''            if cfg.get("actions"):
                outs, results, alog, said = run_actions(rep.raw, st, outs)''',
    '''            if cfg.get("actions"):
                outs, results, alog, said = run_actions(rep.raw, st, outs)
                if said:
                    st.last_write_tick = t''')
sub('''        if cfg.get("actions"):
            rec["actions"] = alog if not ((rep.error and not rep.raw) or refused) else []''',
    '''        if cfg.get("actions"):
            rec["actions"] = alog if not ((rep.error and not rep.raw) or refused) else []
            if cfg.get("say_only"):
                rec["out_types"] = list(st.out_types); rec["last_write_tick"] = st.last_write_tick''')
sub('''def read_inputs(world: list[dict], states: list[AgentState], t: int | None = None,''',
    '''def bridge_intake(agent: dict, states: list[AgentState], t: int) -> list:
    """per input port: (decoded value from the sender's declared type, sender busy = wrote nothing last tick)."""
    out = []
    for l in agent["links"]:
        if not l:
            out.append(("empty", False)); continue
        src = states[l["agent"]]
        out.append((port_decode(src.outputs[l["port"]], src.out_types[l["port"]]), t > 1 and src.last_write_tick < t - 1))
    return out


def read_inputs(world: list[dict], states: list[AgentState], t: int | None = None,''')

# 11. resume: types and write ticks come back
sub('''            st.action_results = list(a.get("action_results", []))''',
    '''            st.action_results = list(a.get("action_results", []))
            st.out_types = list(a.get("out_types", st.out_types)); st.last_write_tick = int(a.get("last_write_tick", 0))''')

# 12. validation: the bridge is the only channel
sub('''    if cfg.get("say_only") and not cfg.get("actions"):
        sys.exit("config: say_only needs actions = true, or nothing can write a port")''',
    '''    if cfg.get("say_only") and not cfg.get("actions"):
        sys.exit("config: say_only needs actions = true, or nothing can write a port")
    if cfg.get("say_only") and registers.on(cfg):
        sys.exit("config: say_only (the bridge) and registers together give the nodes a second channel; pick one")''')

p.write_text(s, encoding='utf-8')
print('patched sim.py')
