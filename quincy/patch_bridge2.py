#!/usr/bin/env python3
"""Bridge, part 2 (owner 2026-11-09: "have we given them any way to save enough state to coordinate?"): the host keeps
state for them. (1) a ledger per input port of every decoded value that arrived, with the tick, shown every tick; (2) a notes
table set with <note key="k">v</note> (empty value forgets), shown every tick. Both are recorded per tick and restored on resume.
Run after patch_bridge.py from the fourfold root."""
import sys
from pathlib import Path

p = Path('sim.py'); s = p.read_text(encoding='utf-8')
if 'out_types' not in s:
    sys.exit('run patch_bridge.py first')
if 'ledger' in s:
    sys.exit('already patched')


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f'expected {count} of: {old[:70]!r}, found {s.count(old)}'
    s = s.replace(old, new)


sub('''    last_write_tick: int = 0                         # bridge: the last tick this agent wrote a port (0 = never)\n''',
    '''    last_write_tick: int = 0                         # bridge: the last tick this agent wrote a port (0 = never)
    ledger: list = field(default_factory=lambda: [[] for _ in range(4)])   # bridge: per input port, [(tick, decoded)] as values arrived
    notes: dict = field(default_factory=dict)        # bridge: the agent's own key -> value table, shown every tick\n''')

# the note action
sub('''    r'|<compute>(?P<expr>.*?)</compute>', re.S | re.I)''',
    '''    r'|<compute>(?P<expr>.*?)</compute>'
    r'|<note\\s+key\\s*=\\s*["\\'](?P<nkey>[^"\\']{1,24})["\\']\\s*>(?P<nval>.*?)</note>', re.S | re.I)
MAX_NOTES = 32''')
sub('''        elif m.group("expr") is not None:
            expr = " ".join(m.group("expr").split())''',
    '''        elif m.group("nkey") is not None:
            key, val = m.group("nkey").strip(), " ".join(m.group("nval").split())[:64]
            if not val:
                st.notes.pop(key, None); results.append(f"note {key}: forgotten"); log.append({"kind": "note", "key": key, "value": ""})
            elif key in st.notes or len(st.notes) < MAX_NOTES:
                st.notes[key] = val; results.append(f"note {key} = {val}"); log.append({"kind": "note", "key": key, "value": val})
            else:
                results.append(f"note {key} rejected: you already have {MAX_NOTES} notes; forget one first (an empty value)."); log.append({"kind": "note", "key": key, "rejected": "full"})
        elif m.group("expr") is not None:
            expr = " ".join(m.group("expr").split())''')

# the ledger: updated where the intake is built, shown with the notes under the ports
sub('''def bridge_intake(agent: dict, states: list[AgentState], t: int) -> list:
    """per input port: (decoded value from the sender's declared type, sender busy = wrote nothing last tick)."""
    out = []
    for l in agent["links"]:
        if not l:
            out.append(("empty", False)); continue
        src = states[l["agent"]]
        out.append((port_decode(src.outputs[l["port"]], src.out_types[l["port"]]), t > 1 and src.last_write_tick < t - 1))
    return out''',
    '''def bridge_intake(agent: dict, states: list[AgentState], t: int) -> list:
    """per input port: (decoded value from the sender's declared type, sender busy = wrote nothing last tick).
    also appends every new non-empty decoded value to the reader's ledger, with the tick it arrived."""
    out, me = [], states[agent["id"]]
    for i, l in enumerate(agent["links"]):
        if not l:
            out.append(("empty", False)); continue
        src = states[l["agent"]]
        decoded = port_decode(src.outputs[l["port"]], src.out_types[l["port"]])
        out.append((decoded, t > 1 and src.last_write_tick < t - 1))
        if decoded not in ("empty", "unknown") and (not me.ledger[i] or me.ledger[i][-1][1] != decoded):
            me.ledger[i].append((t, decoded))
    return out


LEDGER_SHOWN = 8


def bridge_state_block(st: AgentState) -> str:
    """what the host remembers for the agent: the ledger of arrivals per port and its notes table."""
    lines = ["\\narrived on your input ports so far (tick: value, oldest first, last %d per port):" % LEDGER_SHOWN]
    for i in range(4):
        rows = st.ledger[i][-LEDGER_SHOWN:]
        lines.append(f"p{i}: " + ("; ".join(f"t{tt}: {d}" for tt, d in rows) if rows else "nothing yet"))
    lines.append("\\nyour notes (key = value):")
    lines.extend(f"{k} = {v}" for k, v in st.notes.items()) if st.notes else lines.append("(none; set one with <note key=\\"k\\">value</note>)")
    return "\\n".join(lines) + "\\n"''')

# show the block under the bridge
sub('''                   + (action_blocks(st, inp, bridge=bool(cfg.get("say_only"))) if cfg.get("actions") else ""),''',
    '''                   + (bridge_state_block(st) if cfg.get("say_only") else "")
                   + (action_blocks(st, inp, bridge=bool(cfg.get("say_only"))) if cfg.get("actions") else ""),''')

# record and restore
sub('''            if cfg.get("say_only"):
                rec["out_types"] = list(st.out_types); rec["last_write_tick"] = st.last_write_tick''',
    '''            if cfg.get("say_only"):
                rec["out_types"] = list(st.out_types); rec["last_write_tick"] = st.last_write_tick
                rec["ledger"] = [list(map(list, x)) for x in st.ledger]; rec["notes"] = dict(st.notes)''')
sub('''            st.out_types = list(a.get("out_types", st.out_types)); st.last_write_tick = int(a.get("last_write_tick", 0))''',
    '''            st.out_types = list(a.get("out_types", st.out_types)); st.last_write_tick = int(a.get("last_write_tick", 0))
            st.ledger = [[tuple(x) for x in port] for port in a.get("ledger", st.ledger)]; st.notes = dict(a.get("notes", {}))''')

p.write_text(s, encoding='utf-8')
print('patched sim.py (ledger + notes)')
