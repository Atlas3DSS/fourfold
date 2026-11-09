#!/usr/bin/env python3
"""eval.py: FOURFOLD_EVAL_BASE_URL overrides the run's llm base_url, so a finished run can be evaluated on another live server
when the one it ran on is gone (fabe-integration, 2026-11-09). Run from the fourfold root; idempotent."""
import sys
from pathlib import Path

p = Path("eval.py"); s = p.read_text(encoding="utf-8")
if "FOURFOLD_EVAL_BASE_URL" in s:
    sys.exit("already patched")
old = '    agent = sim.make_agent(meta["config"])'
assert s.count(old) == 1, s.count(old)
new = '''    if os.environ.get("FOURFOLD_EVAL_BASE_URL"):      # fabe 2026-11-09: evaluate on another live server when the original is gone
        meta["config"]["llm"]["base_url"] = [os.environ["FOURFOLD_EVAL_BASE_URL"]]
        meta["config"]["llm"].pop("pools", None)
        print("eval: base_url overridden ->", meta["config"]["llm"]["base_url"], file=sys.stderr)
    agent = sim.make_agent(meta["config"])'''
s = s.replace(old, new)
for mod in ("os", "sys"):
    if f"import {mod}" not in s:
        s = s.replace("import argparse", f"import argparse\nimport {mod}", 1)
p.write_text(s, encoding="utf-8")
print("eval.py patched")
