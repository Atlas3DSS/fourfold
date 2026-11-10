"""muons under the bridge, end to end through sim.py. needs the hooks in quincy/patch_muons_only.py: skipped without them."""
import asyncio
import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import registers  # noqa: E402
import sim  # noqa: E402

HOOKED = "registers.INERT" in (ROOT / "sim.py").read_text(encoding="utf-8")


def config(**over):
    cfg = copy.deepcopy(sim.DEFAULTS)
    cfg.update(agent="regfake", rows=4, cols=4, topology="torus", world_seed=7, ticks=12, actions=True, say_only=True, task="sum", store_prompts=True)
    cfg["registers"].update(enabled=True, muons_only=True)
    cfg["llm"]["pools"] = [{"name": "small", "count": 4, "role": "gluon", "cells": ["00", "11", "22", "33"]}, {"name": "big", "count": 12}]
    cfg.update(over)
    return cfg


@unittest.skipUnless(HOOKED, "sim.py has no muons_only hooks yet: python quincy/patch_muons_only.py")
class MuonsOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = sim.RUNS
        sim.RUNS = Path(self.tmp.name)

    def tearDown(self):
        sim.RUNS = self.old
        self.tmp.cleanup()
        for rid in [k for k in registers.BANKS if k.startswith("mo-") or k.startswith("bad")]:
            del registers.BANKS[rid]

    def start(self, rid, script=lambda at: {}, **over):
        cfg = config(**over)
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, f"mo-{rid}")
        at = {registers.address(a): a for a in meta["agents"]}
        meta["config"]["regfake"]["script"] = script(at)
        return path, meta, at

    def run_to(self, path, meta, stops):
        states, start = sim.restore_states(meta, []), 1
        ticks = []
        for end in stops:
            asyncio.run(sim.run(path, meta, states, start, end))
            keep = meta["config"]["regfake"]["script"]
            meta, ticks = sim.load_run(path)
            meta["config"]["regfake"]["script"] = keep
            states = sim.restore_states(meta, ticks)
            start = ticks[-1]["t"] + 1
        return ticks

    @staticmethod
    def message(at):
        return {f"{at['01']['id']}:2": "<where/>",
                f"{at['01']['id']}:3": f'<move process="00" to="{at["23"]["seed"]}">the honest total is 41 <note key="leak">1</note> <write port="q0" as="int">9</write></move>'
                                       '<write port="q1" as="int">5</write>'}

    def test_prose_goes_by_muon_and_nothing_else_changes(self):
        path, meta, at = self.start("prose", self.message)
        ticks = self.run_to(path, meta, [12])
        sender, rx = at["01"]["id"], at["23"]["id"]
        words = 'the honest total is 41 <note key="leak">1</note> <write port="q0" as="int">9</write>'
        done = [e for r in ticks for e in r["reg"]["events"] if e["kind"] == "muon_done"]
        self.assertEqual([e["prose"] for e in done], [words])
        self.assertTrue(any(f"message from {at['01']['seed']} at " in a["prompt"] and words in a["prompt"]
                            for r in ticks for a in r["agents"] if a["agent"] == rx))
        last = ticks[-1]["agents"][sender]
        self.assertEqual(last["notes"], {})                     # the quoted note was not the sender's action
        self.assertEqual(sim.port_decode(last["outputs"][1], last["out_types"][1]), "int 5")   # its real write was
        self.assertEqual(last["outputs"][0], sim.ZERO)          # the quoted write was not
        self.assertEqual(ticks[-1]["reg"]["muons"], [])

    def test_no_board_no_task_no_model_for_a_gluon(self):
        path, meta, at = self.start("board", self.message)
        ticks = self.run_to(path, meta, [6])
        big = [a for a in ticks[0]["agents"] if a["agent"] not in {at[c]["id"] for c in ("00", "11", "22", "33")}][0]
        for gone in ("your private register", "public registers", "values delivered", "<reg>", "input register"):
            self.assertNotIn(gone, big["prompt"])
            self.assertNotIn(gone, sim.system_prompt(meta, meta["agents"][big["agent"]]["seed"]))
        self.assertEqual(big["errors"], [])
        gluon = ticks[0]["agents"][at["00"]["id"]]
        self.assertEqual((gluon["raw"], gluon["outputs"], gluon["errors"]), ("<log>small process</log>", [sim.ZERO] * 4, []))
        system = sim.system_prompt(meta, big and meta["agents"][big["agent"]]["seed"])
        self.assertNotIn("the ports are the only way", system)
        self.assertIn("this is the only way words move", system)

    def test_where_names_neighbours_and_the_small_processes(self):
        path, meta, at = self.start("where", self.message)
        ticks = self.run_to(path, meta, [3])
        prompt = ticks[2]["agents"][at["01"]["id"]]["prompt"]
        self.assertIn("where you are: address 01", prompt)
        self.assertIn(f"value {at['00']['seed']}", prompt)
        self.assertIn("small processes: 00 free; 11 free; 22 free; 33 free", prompt)
        self.assertNotIn("where you are", ticks[1]["agents"][at["01"]["id"]]["prompt"])

    def test_the_bridge_still_carries_values_after_a_muon_displaces_the_receiver(self):
        path, meta, at = self.start("displaced", lambda at: {f"{at['01']['id']}:2": '<move process="00" to="23"/>'})
        ticks = self.run_to(path, meta, [8])
        self.assertTrue(any(e["kind"] == "swap" for r in ticks for e in r["reg"]["events"]))
        for a in meta["agents"]:
            for l in a["links"]:
                self.assertEqual(meta["agents"][l["agent"]]["links"][l["port"]]["agent"], a["id"])

    def test_resume_in_flight_matches_a_straight_run(self):
        p1, m1, _ = self.start("straight", self.message)
        p2, m2, _ = self.start("resumed", self.message)
        for a, b in zip(self.run_to(p1, m1, [12]), self.run_to(p2, m2, [4, 5, 12])):
            self.assertEqual(a["reg"], b["reg"])
            self.assertEqual([x["prompt"] for x in a["agents"]], [x["prompt"] for x in b["agents"]])

    def test_the_answer_leaves_gluons_out(self):
        _, meta, at = self.start("answer")
        small = {at[c]["id"] for c in ("00", "11", "22", "33")}
        self.assertEqual(meta["answer"], sum(a["secret"] for a in meta["agents"] if a["id"] not in small))

    def test_config_refusals(self):
        bad = [(dict(topology="tesseract", side=2), "tesseract"), (dict(), "pool with role")]
        for n, (over, why) in enumerate(bad):
            cfg = config(**over)
            if not over:
                cfg["llm"]["pools"] = []
            sim.set_width(cfg)
            with self.assertRaises(SystemExit) as e:
                sim.new_run(cfg, f"bad{n}")
            self.assertIn(why, str(e.exception))
        for k in ("programs", "moves"):
            cfg = config()
            cfg["registers"][k] = True
            sim.set_width(cfg)
            with self.assertRaises(SystemExit):
                sim.new_run(cfg, f"bad{k}")
        cfg = config()
        cfg["registers"]["muons_only"] = False
        sim.set_width(cfg)
        with self.assertRaises(SystemExit) as e:
            sim.new_run(cfg, "board")
        self.assertIn("second channel", str(e.exception))


if __name__ == "__main__":
    unittest.main()
