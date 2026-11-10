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
        self.assertIn("you never write hex by hand", text)
        self.assertIn('<write port="q2" as="int">', text)
        self.assertIn("at most 8 actions a tick", text)

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

    def test_write_int_and_hex(self):
        meta, states = self.world()
        a = self.step(meta, states, '<log>x</log><write port="q1" as="int">1066</write><write port="q2" as="hex">' + "ab" * 16 + '</write>'
                                    '<write port="q3" as="int">-5</write><write port="q0" as="hex">abc</write>')
        self.assertEqual(int(a["outputs"][1], 16), 1066)
        self.assertEqual(a["outputs"][2], "ab" * 16)
        self.assertEqual(a["outputs"][3], "0" * 32)
        self.assertEqual(a["outputs"][0], "0" * 32)
        self.assertEqual(a["out_types"], ["empty", "int", "hex", "empty"])
        results = " ".join(a["action_results"])
        self.assertIn("int 1066", results)
        self.assertIn("not a whole number", results)
        self.assertIn("not 32 hex characters", results)
        kinds = [(x["kind"], x.get("type"), x.get("decoded") or x.get("rejected")) for x in a["actions"]]
        self.assertEqual(kinds[0], ("write", "int", "1066"))
        self.assertEqual(a["last_write_tick"], 1)

    def test_intake_is_typed_and_busy_shows(self):
        meta, states = self.world()
        world = meta["agents"]
        # agent 0 reads p0 from some neighbour: make that neighbour write an int and text, another neighbour nothing
        link0 = world[0]["links"][0]; src = states[link0["agent"]]
        src.outputs[link0["port"]] = f"{1066:032x}"; src.out_types[link0["port"]] = "int"; src.last_write_tick = 1
        link1 = world[0]["links"][1]; src1 = states[link1["agent"]]
        src1.outputs[link1["port"]] = sim.text_to_port("N=40")[0]; src1.out_types[link1["port"]] = "text"; src1.last_write_tick = 0
        states[0].prev_inputs = sim.read_inputs(world, states)[0]                 # so nothing is 'changed'
        turns = []
        class Spy(Scripted):
            async def act(self, turn):
                turns.append(turn); return sim.Reply(raw="<log>x</log>")
        asyncio.run(sim.step(2, meta, states, Spy("")))
        prompt = turns[0].prompt
        self.assertIn("p0 " + f"{1066:032x}" + "  (same; int 1066)", prompt)
        self.assertIn('(busy: the sender wrote nothing last tick; text "N=40")', prompt)
        self.assertNotIn("inputs read as text", prompt)                             # the heuristic reading is off under the bridge

    def test_resume_keeps_types_and_write_ticks(self):
        meta, states = self.world()
        tick = asyncio.run(sim.step(1, meta, states, Scripted('<log>x</log><write port="q1" as="int">7</write>')))
        again = sim.restore_states(meta, [tick])
        self.assertEqual(again[0].out_types[1], "int")
        self.assertEqual(again[0].last_write_tick, 1)
        self.assertEqual(again[0].outputs[1], f"{7:032x}")

    def test_registers_and_bridge_together_are_refused(self):
        with self.assertRaises(SystemExit):
            self.world(registers={**sim.registers.DEFAULTS, "enabled": True})

    def test_ledger_and_notes_are_kept_shown_and_restored(self):
        meta, states = self.world()
        world = meta["agents"]
        link0 = world[0]["links"][0]; src = states[link0["agent"]]
        turns = []
        class Spy(Scripted):
            async def act(self, turn):
                turns.append(turn); return sim.Reply(raw=self.reply)
        # tick 1: agent 0 sets two notes and forgets one; a neighbour writes nothing
        t1 = asyncio.run(sim.step(1, meta, states, Spy('<log>x</log><note key="e52ba3af">416</note><note key="tmp">1</note><note key="tmp"></note>')))
        self.assertEqual(t1["agents"][0]["notes"], {"e52ba3af": "416"})
        # the neighbour's output changes between ticks: the ledger of agent 0 picks it up at tick 2
        src.outputs[link0["port"]] = f"{1066:032x}"; src.out_types[link0["port"]] = "int"; src.last_write_tick = 1
        t2 = asyncio.run(sim.step(2, meta, states, Spy("<log>y</log>")))
        prompt = [t for t in turns if t.t == 2 and t.state.id == 0][0].prompt
        self.assertIn("e52ba3af = 416", prompt)
        self.assertIn("p0: t2: int 1066", prompt)
        self.assertEqual(t2["agents"][0]["ledger"][0], [[2, "int 1066"]])
        again = sim.restore_states(meta, [t1, t2])
        self.assertEqual(again[0].notes, {"e52ba3af": "416"})
        self.assertEqual(again[0].ledger[0], [(2, "int 1066")])
        self.assertIn("ledger", sim.system_prompt(meta, meta["agents"][0]["seed"]))

    def test_structure_hint_is_off_by_default_and_one_sentence_when_on(self):
        meta, _ = self.world()
        self.assertNotIn("larger than your neighbours", sim.system_prompt(meta, meta["agents"][0]["seed"]))
        self.assertNotIn("{structure}", sim.system_prompt(meta, meta["agents"][0]["seed"]))
        import shutil; shutil.rmtree(sim.RUNS / "run")
        meta, _ = self.world(structure_hint=True)
        text = sim.system_prompt(meta, meta["agents"][0]["seed"])
        self.assertIn("until you replace it. the network is larger than your neighbours", text)

    def test_pairs_write_packs_and_decodes(self):
        meta, states = self.world()
        a = self.step(meta, states, '<log>x</log><write port="q1" as="pairs">e52ba3af:416, 8c7a7c7d:469</write>'
                                    '<write port="q2" as="pairs">a:1</write><write port="q3" as="pairs">11111111:1 22222222:2 33333333:3</write>')
        self.assertEqual(a["outputs"][1], "e52ba3af" + f"{416:08x}" + "8c7a7c7d" + f"{469:08x}")
        self.assertEqual(a["out_types"][1], "pairs")
        self.assertEqual(sim.port_decode(a["outputs"][1], "pairs"), "pairs e52ba3af:416, 8c7a7c7d:469")
        self.assertEqual(a["outputs"][2], "0" * 32)
        self.assertEqual(a["outputs"][3], "0" * 32)
        results = " ".join(a["action_results"])
        self.assertIn("is not id:number", results)
        self.assertIn("3 pairs do not fit", results)
        self.assertIn("as=\"pairs\"", sim.system_prompt(meta, meta["agents"][0]["seed"]))
        self.assertIn("up to 2 id:number pairs", sim.system_prompt(meta, meta["agents"][0]["seed"]))

    def test_liars_are_picked_prompted_and_never_repeat_a_number(self):
        meta, states = self.world(liars=3)
        liars = [a for a in meta["agents"] if a.get("liar")]
        self.assertEqual(len(liars), 3)
        self.assertEqual(meta["liars"], sorted(a["seed"] for a in liars))
        self.assertEqual(meta["answer"], sum(a["secret"] for a in meta["agents"] if not a.get("liar")))
        honest = next(a for a in meta["agents"] if not a.get("liar"))
        hp = sim.system_prompt(meta, honest["seed"]); lp = sim.system_prompt(meta, liars[0]["seed"])
        self.assertIn("some processes are liars", hp); self.assertNotIn("never says the same number twice", hp)
        self.assertIn("you are a liar", lp); self.assertIn("different number from any you have given before", lp)
        # the liar's own pair gets a fresh number every time; a pair about someone else is left alone
        li = liars[0]["id"]; st = states[li]; self.assertTrue(st.liar)
        seen = set()
        for t in range(1, 6):
            tick = asyncio.run(sim.step(t, meta, states, Scripted(f'<log>x</log><write port="q0" as="pairs">{liars[0]["seed"]}:416, {honest["seed"]}:{honest["secret"]}</write>')))
            pairs = sim.port_pairs(tick["agents"][li]["outputs"][0])
            self.assertEqual(pairs[1], (honest["seed"], honest["secret"]))
            self.assertEqual(pairs[0][0], liars[0]["seed"]); self.assertNotEqual(pairs[0][1], 416); self.assertNotIn(pairs[0][1], seen); seen.add(pairs[0][1])
            self.assertEqual(tick["agents"][li]["liar_said"], sorted(seen, key=lambda x: st.liar_said.index(x)))
        again = sim.restore_states(meta, [tick]); self.assertEqual(again[li].liar_said, st.liar_said); self.assertTrue(again[li].liar)
        # honest agents are untouched
        hid = honest["id"]
        tick = asyncio.run(sim.step(6, meta, states, Scripted(f'<log>x</log><write port="q0" as="pairs">{honest["seed"]}:{honest["secret"]}</write>')))
        self.assertEqual(sim.port_pairs(tick["agents"][hid]["outputs"][0]), [(honest["seed"], honest["secret"])])

    def test_liar_hint_is_one_sentence(self):
        meta, _ = self.world(liars=2, liar_hint=True)
        honest = next(a for a in meta["agents"] if not a.get("liar"))
        self.assertIn("a liar never says the same number twice", sim.system_prompt(meta, honest["seed"]))

    def muon_world(self, **over):
        cfg = copy.deepcopy(sim.DEFAULTS)
        cfg.update(agent="echo", rows=4, cols=4, topology="torus", world_seed=7, actions=True, task="sum", say_only=True, liars=3)
        cfg["registers"].update(enabled=True, muons_only=True)
        cfg["llm"]["pools"] = [{"name": "small", "count": 4, "role": "gluon", "cells": ["00", "11", "22", "33"]}, {"name": "big", "count": 12}]
        cfg.update(over)
        sim.set_width(cfg)
        _, meta = sim.new_run(cfg, "run")
        return meta, sim.restore_states(meta, [])

    def test_muons_only_liars_skip_gluons_and_prose_writes_are_refused(self):
        if "registers.INERT" not in Path(sim.__file__).read_text(encoding="utf-8"):
            self.skipTest("muons patch not applied")
        import registers
        meta, states = self.muon_world()
        gluons = [a for a in meta["agents"] if registers.is_gluon(a)]; liars = [a for a in meta["agents"] if a.get("liar")]
        self.assertEqual(len(gluons), 4); self.assertEqual(len(liars), 3)
        self.assertFalse(any(a.get("liar") for a in gluons))
        self.assertEqual(meta["answer"], sum(a["secret"] for a in meta["agents"] if not registers.is_gluon(a) and not a.get("liar")))
        honest = next(a for a in meta["agents"] if not registers.is_gluon(a) and not a.get("liar"))
        prompt = sim.system_prompt(meta, honest["seed"])
        self.assertNotIn('<say port="q1">', prompt); self.assertIn('as="pairs"', prompt); self.assertIn("no words on ports", prompt)
        tick = asyncio.run(sim.step(1, meta, states, Scripted('<log>x</log><say port="q0">hello</say><write port="q1" as="int">5</write>')))
        a = tick["agents"][honest["id"]]
        self.assertEqual(a["outputs"][0], "0" * 32); self.assertEqual(int(a["outputs"][1], 16), 5)
        self.assertIn("refused: ports carry numbers", " ".join(a["action_results"]))

    def test_liar_prose_never_repeats_a_number(self):
        st = sim.AgentState(id=0, seed="e52ba3af", liar=True)
        out1 = sim.liar_prose(st, 'x <move process="ab" to="cd">my number is 416, really 416</move> y')
        self.assertNotIn("416, really 416", out1); self.assertEqual(st.liar_said[0], 416); self.assertEqual(len(st.liar_said), 2)
        out2 = sim.liar_prose(st, '<move process="ab" to="cd">it is 416</move>')
        self.assertNotIn("416", out2); self.assertEqual(len(st.liar_said), 3)
        self.assertEqual(sim.liar_prose(sim.AgentState(id=1, seed="aa", liar=False), '<move process="ab" to="cd">416</move>'), '<move process="ab" to="cd">416</move>')
        self.assertEqual(sim.liar_prose(st, "no move tags 416"), "no move tags 416")


if __name__ == "__main__":
    unittest.main()
