#!/usr/bin/env python3
"""sim.py hooks for registers.muons_only: muons under the bridge. run from the fourfold root: python quincy/patch_muons_only.py
each replacement must match exactly once or nothing is written. with the flag off, every hook leaves behaviour as it was."""
from pathlib import Path

p = Path("sim.py")
s = p.read_text(encoding="utf-8")
edits = [
    # 1. the bridge and the registers board stay refused together; muons under the bridge are allowed
    ('''    if cfg.get("say_only") and registers.on(cfg):
        sys.exit(''', '''    if cfg.get("say_only") and registers.has_board(cfg):
        sys.exit('''),
    # 2. a gluon's turn runs no model
    ('''        if bank and registers.is_gluon(world[st.id]):
            turns.append(Turn(''', '''        if bank and registers.is_gluon(world[st.id]) and registers.muons_only(cfg):
            turns.append(Turn(t=t, state=st, inputs=inp, changed=changed, system=registers.INERT, prompt=""))
            continue
        if bank and registers.is_gluon(world[st.id]):
            turns.append(Turn('''),
    ('''async def safe_act(agent, turn: Turn) -> Reply:
''', '''async def safe_act(agent, turn: Turn) -> Reply:
    if turn.system == registers.INERT:
        return Reply(raw="<log>small process</log>")
'''),
    # 3. words in a muon's message are not the sender's actions, notes or log
    ('''            outs, lines, proto, perr, notices = parse_response(rep.raw, st.outputs, st.protocol,
''', '''            acted = registers.without_prose(rep.raw) if registers.muons_only(cfg) else rep.raw
            outs, lines, proto, perr, notices = parse_response(acted, st.outputs, st.protocol,
'''),
    ('''                outs, results, alog, said = run_actions(rep.raw, st, outs)
''', '''                outs, results, alog, said = run_actions(acted, st, outs)
'''),
    # 4. the ratchet's board signal means nothing without a board
    ('''    if registers.on(meta["config"]):
        reg, reg_sig = registers.interesting(window, before, n)
''', '''    if registers.has_board(meta["config"]):
        reg, reg_sig = registers.interesting(window, before, n)
'''),
    # 5. the prompt must not say the ports are the only way
    ('''        template = template.replace("{structure}", structure)
''', '''        template = template.replace("{structure}", structure)
        if registers.muons_only(meta["config"]):
            template = template.replace("the ports are the only way processes reach each other. nothing you write\\noutside an action is seen by anyone but the evaluator.",
                                        "the ports carry numbers and short values. words go by the small processes described below. nothing\\nyou write outside an action is seen by anyone but the evaluator.")
'''),
    # 6. gluons don't play: the answer is over the nodes that think
    ('''    registers.assign_pools(meta)
    d = RUNS / run_id
''', '''    registers.assign_pools(meta)
    if registers.muons_only(cfg) and "answer" in meta:
        meta["answer"] = sum(a["secret"] for a in meta["agents"] if not registers.is_gluon(a))
    d = RUNS / run_id
'''),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"no exact single match, nothing written: {old[:70]!r}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print(f"{len(edits)} hooks in")
