"""tests for the say_only flag in sim.py: python -m unittest tests.test_say_only"""
import asyncio
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sim  # noqa: E402


class Scripted:
    def __init__(self, reply):
        self.reply = reply

    async def act(self, turn):
        return sim.Reply(raw=self.reply)

    async def close(self):
        pass


class SayOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = sim.RUNS
        sim.RUNS = Path(self.tmp.name)

    def tearDown(self):
        sim.RUNS = self.old
        self.tmp.cleanup()

    def world(self, **over):
        cfg = copy.deepcopy(sim.DEFAULTS)
        cfg.update(agent="echo", rows=4, cols=4, topology="torus", world_seed=7, actions=True, task="sum", say_only=True)
        cfg.update(over)
        sim.set_width(cfg)
        _, meta = sim.new_run(cfg, "run")
        return meta, sim.restore_states(meta, [])

    def step(self, meta, states, reply):
        return asyncio.run(sim.step(1, meta, states, Scripted(reply)))["agents"][0]

    def test_format_has_no_port_tags(self):
        meta, _ = self.world()
        text = sim.system_prompt(meta, meta["agents"][0]["seed"])
        self.assertNotIn("<q0>", text)
        self.assertNotIn("{fmt}", text)
        self.assertIn("<log>...</log>", text)
        self.assertIn("only be written with say and ask", text)

    def test_hand_written_tags_write_nothing(self):
        meta, states = self.world()
        too_long = "494420616e64206e756d3f0000000000000"
        a = self.step(meta, states, f"<log>x</log><q0>{too_long}</q0><q1>{'a' * 32}</q1>")
        self.assertEqual(a["outputs"], ["0" * 32] * 4)
        self.assertEqual(a["errors"], ["q tags ignored"])
        self.assertIn("only <say> and <ask>", a["notices"][0])

    def test_say_writes_and_a_clean_reply_has_no_errors(self):
        meta, states = self.world()
        a = self.step(meta, states, '<log>x</log><say port="q2">417 from me</say>')
        self.assertEqual(bytes.fromhex(a["outputs"][2]).strip(b"\x00").decode(), "417 from me")
        self.assertEqual(a["errors"], [])
        self.assertEqual(a["notices"], [])
        self.assertEqual(len(a["outputs"][2]), 32)

    def test_say_beside_a_tag_keeps_the_say(self):
        meta, states = self.world()
        a = self.step(meta, states, f'<log>x</log><say port="q0">hi there</say><q0>{"b" * 32}</q0><q1>{"c" * 32}</q1>')
        self.assertTrue(a["outputs"][0].startswith("hi there".encode().hex()))
        self.assertEqual(a["outputs"][1], "0" * 32)

    def test_nonsense_gets_the_plain_notice_and_resume_keeps_outputs(self):
        meta, states = self.world()
        self.assertEqual(self.step(meta, states, "hmm")["notices"], [sim.NO_REPLY_NOTICE])

    def test_needs_actions(self):
        with self.assertRaises(SystemExit):
            self.world(actions=False)

    def test_off_by_default_tags_still_work(self):
        meta, states = self.world(say_only=False)
        a = self.step(meta, states, f"<log>x</log><q0>{'a' * 32}</q0><q1>{'b' * 32}</q1><q2>{'c' * 32}</q2><q3>{'d' * 32}</q3>")
        self.assertEqual(a["outputs"], ["a" * 32, "b" * 32, "c" * 32, "d" * 32])
        self.assertEqual(a["errors"], [])
        self.assertIn("<q0>", sim.system_prompt(meta, meta["agents"][0]["seed"]))


if __name__ == "__main__":
    unittest.main()
