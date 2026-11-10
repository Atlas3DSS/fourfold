"""tests for the tree and ring typed writes and the scripted agents: python -m unittest tests.test_tree_ring"""
import asyncio
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sim  # noqa: E402


class TreeRing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = sim.RUNS
        sim.RUNS = Path(self.tmp.name)

    def tearDown(self):
        sim.RUNS = self.old
        self.tmp.cleanup()

    def world(self, **over):
        cfg = copy.deepcopy(sim.DEFAULTS)
        cfg.update(agent="scripted", rows=4, cols=4, topology="torus", world_seed=7, actions=True, task="sum", say_only=True)
        cfg.update(over)
        sim.set_width(cfg)
        _, meta = sim.new_run(cfg, "run")
        return meta, sim.restore_states(meta, [])

    def run_to(self, meta, states, ticks):
        agent = sim.make_agent(meta["config"])
        return [asyncio.run(sim.step(t, meta, states, agent)) for t in range(1, ticks + 1)]

    def test_tree_write_adds_exactly_and_keeps_the_other_tree(self):
        self.world()
        v, why, ent = sim.tree_to_port("A sub 417 + 1200+88 1+5+2", sim.ZERO, "empty")
        self.assertEqual(ent, {"A": ("sub", 1705, 8)})
        v2, _, ent2 = sim.tree_to_port("B wave", v, "tree")
        self.assertEqual(ent2, {"A": ("sub", 1705, 8), "B": ("wave", 0, 0)})
        self.assertEqual(sim.port_decode(v2, "tree"), "tree A sub 1705 8; B wave")
        self.assertEqual(sim.tree_to_port("B wave", v, "pairs")[2], {"B": ("wave", 0, 0)})   # another type is replaced

    def test_tree_write_rejects_nonsense(self):
        self.world()
        for bad in ("C wave", "A sub 5", "A wave 3 1", "A total 5 0", "hello", ""):
            self.assertIsNone(sim.tree_to_port(bad, sim.ZERO, "empty")[0], bad)

    def test_ring_write(self):
        self.world()
        v, _, ent = sim.ring_to_port("8c7a7c7d 469+417 1+1; e52ba3af 416 1")
        self.assertEqual(ent, [("8c7a7c7d", 886, 2), ("e52ba3af", 416, 1)])
        self.assertEqual(sim.port_decode(v, "ring"), "ring 8c7a7c7d 886 2; e52ba3af 416 1")
        for bad in ("8c7a7c7d 5", "8c7a7c7d 1048576 1", "8c7a7c7d 5 4096", "a 1 1; b 2 1; c 3 1", "8c7a7c7d 1 1; 8c7a7c7d 2 1"):
            self.assertIsNone(sim.ring_to_port(bad)[0], bad)

    def test_ring_none_clears_the_port(self):
        self.world()
        self.assertEqual(sim.ring_to_port("none")[0], sim.ZERO)
        self.assertEqual(sim.port_decode(sim.ZERO, "ring"), "ring none")

    def test_wave_on_the_parent_output_is_refused(self):
        self.world()
        wave = sim.tree_to_port("B wave", sim.ZERO, "empty")[2]
        v, why, _ = sim.tree_to_port("A wave; B wave", sim.ZERO, "empty", [wave, wave, wave], "")
        self.assertIsNone(v)
        self.assertIn("parent's output for tree B", why)
        self.assertIsNotNone(sim.tree_to_port("A wave", sim.ZERO, "empty", [wave, wave, wave], "")[0])           # the other tree is free
        self.assertIsNotNone(sim.tree_to_port("B wave", sim.ZERO, "empty", [wave, wave, wave], "B")[0])          # the root waves on all four
        self.assertIsNotNone(sim.tree_to_port("B sub 5 1", sim.ZERO, "empty", [wave, wave, wave], "")[0])

    def test_ring_write_is_checked_by_the_host(self):
        meta, states = self.world(rows=4, cols=4, topology="tesseract", side=2, protocol_hint="ring")
        st = states[0]
        st.secret, st.ring_res, st.ring_in = 416, {}, [[("fe9fac43", 411, 1)], [], [], []]
        ok = lambda d, text: sim.ring_check_write(st, d, sim.ring_to_port(text)[2])
        self.assertEqual(ok(0, "fe9fac43 411+416 1+1; e52ba3af 416 1".replace("e52ba3af", st.seed)), "")
        self.assertIn("always your base", ok(0, f"{st.seed} 416+416 1+1"))
        self.assertIn("passed on with your base", ok(0, "fe9fac43 411+827 1+2"))
        self.assertIn("shows no entry", ok(0, "8c7a7c7d 5 1"))
        self.assertIn("no base for ring 1", ok(1, f"{st.seed} 827 2"))
        st.ring_res = {0: (827, 2)}
        self.assertEqual(ok(1, f"{st.seed} 827 2"), "")
        self.assertIn("finished", ok(0, f"{st.seed} 416 1"))
        st.ring_in[1] = [("43703c8a", 907, 2)]
        st.ring_res[1] = (1734, 4)
        self.assertEqual(ok(1, "43703c8a 907+827 2+2"), "")                  # ring 1 keeps the ring-1 base after it is finished
        self.assertIn("passed on with your base", ok(1, "43703c8a 907+1734 2+4"))   # the slip the first 27B smoke made

    def test_pairs_and_int_take_sums(self):
        self.world()
        self.assertEqual(sim.pairs_to_port("8c7a7c7d:469+417")[2], [("8c7a7c7d", 886)])
        self.assertEqual(int(sim.int_to_port("1000 + 66")[0], 16), 1066)
        self.assertEqual(int(sim.int_to_port("1,066")[0], 16), 1066)

    def test_prompt_tells_the_method_and_the_root(self):
        meta, _ = self.world(protocol_hint="tree", tree_roots=[3, 9])
        seeds = [a["seed"] for a in meta["agents"]]
        self.assertIn("you are the root of tree A", sim.system_prompt(meta, seeds[3]))
        self.assertIn("you are the root of tree B", sim.system_prompt(meta, seeds[9]))
        other = sim.system_prompt(meta, seeds[0])
        self.assertNotIn("you are the root", other)
        self.assertIn("both trees start at other processes", other)
        self.assertIn('as="tree"', other)
        sim.RUNS = Path(tempfile.mkdtemp(dir=self.tmp.name))
        plain, _ = self.world(world_seed=8)
        self.assertNotIn('as="tree"', sim.system_prompt(plain, plain["agents"][0]["seed"]))

    def test_scripted_tree_gives_every_agent_the_exact_total(self):
        for roots in ([0], [0, 10]):
            sim.RUNS = Path(tempfile.mkdtemp(dir=self.tmp.name))
            meta, states = self.world(protocol_hint="tree", tree_roots=roots, scripted={"kind": "tree", "miss_rate": 0.0})
            last = self.run_to(meta, states, 14)[-1]
            for a in last["agents"]:
                self.assertTrue(a["protocol"].startswith(f"total {meta['answer']} over 16 processes"), a["protocol"])

    def test_scripted_ring_on_the_tesseract(self):
        meta, states = self.world(rows=9, cols=9, topology="tesseract", side=3, protocol_hint="ring",
                                  scripted={"kind": "ring", "miss_rate": 0.0})
        last = self.run_to(meta, states, 13)[-1]
        for a in last["agents"]:
            self.assertEqual(a["protocol"], f"total {meta['answer']} over 81 processes")

    def test_seam_holds_the_typed_value_too(self):
        meta, states = self.world(rows=4, cols=8, modules=[4, 4], seam_period=2, protocol_hint="tree", tree_roots=[0],
                                  scripted={"kind": "tree", "miss_rate": 0.0})
        ticks = self.run_to(meta, states, 40)
        self.assertTrue(all(a["protocol"] == f"total {meta['answer']} over 32 processes" for a in ticks[-1]["agents"]))


if __name__ == "__main__":
    unittest.main()
