"""tests for fourfold/registers.py and its hooks in sim.py: run with  python -m unittest tests.test_registers"""
import asyncio
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quincy"))
import httpx  # noqa: E402

import registers  # noqa: E402
import reg_copy_test  # noqa: E402
import sim  # noqa: E402


def config(**over):
    cfg = copy.deepcopy(sim.DEFAULTS)
    cfg.update(agent="regfake", rows=4, cols=4, topology="torus", world_seed=7, port_bits=8, ticks=30)
    cfg["registers"].update(enabled=True, tell_cells=True)
    for k, v in over.items():
        if k in registers.DEFAULTS:
            cfg["registers"][k] = v
        else:
            cfg[k] = v
    return cfg


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_runs = sim.RUNS
        sim.RUNS = Path(self.tmp.name)
        self.n = 0

    def tearDown(self):
        sim.RUNS = self.old_runs
        self.tmp.cleanup()

    def world(self, **over):
        self.n += 1
        cfg = config(**over)
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, f"t{self.n}")
        return path, meta, sim.restore_states(meta, [])

    def go(self, meta, states, end, start=1, path=None, ratchet=0):
        path = path or sim.RUNS / meta["run_id"] / "ticks.jsonl"
        asyncio.run(sim.run(path, meta, states, start, end, ratchet))
        return sim.load_run(path)[1]

    def line(self, meta, i, pub="00", priv="0000", extra=""):
        return f"<reg>{registers.address(meta['agents'][i])} {pub} {priv}</reg>\n{extra}"


class Wiring(Base):
    def test_rewire_matches_build_world(self):
        for torus in (True, False):
            for rows, cols in ((3, 3), (4, 4), (3, 5)):
                import random
                agents = sim.build_world(rows, cols, random.Random(1), torus=torus)
                want = copy.deepcopy([a["links"] for a in agents])
                registers.rewire(agents, rows, cols, torus)
                self.assertEqual(want, [a["links"] for a in agents])

    def test_swap_keeps_links_reciprocal_and_ports_personal(self):
        _, meta, _ = self.world(moves=True)
        a, b = meta["agents"][0], next(x for x in meta["agents"] if x["id"] in registers.neighbours(meta["agents"][0]))
        sides = list(a["port_sides"]), list(b["port_sides"])
        a["pos"], b["pos"] = b["pos"], a["pos"]
        registers._rewire_meta(meta)
        self.assertEqual(sides, (a["port_sides"], b["port_sides"]))
        for x in meta["agents"]:
            for i, l in enumerate(x["links"]):
                back = meta["agents"][l["agent"]]["links"][l["port"]]
                self.assertEqual((back["agent"], back["port"]), (x["id"], i))
        self.assertIn(b["id"], registers.neighbours(a))


class Parsing(Base):
    def test_good_line_and_case(self):
        self.assertEqual(registers.parse_reg("<reg>2A 3F 00aB</reg>", "2a", 2, 4), ("3f", "00ab", True, [], []))

    def test_not_first_is_recorded(self):
        self.assertFalse(registers.parse_reg("hello\n<reg>2a 3f 00ab</reg>", "2a", 2, 4)[2])

    def test_missing_and_malformed_hold(self):
        self.assertEqual(registers.parse_reg("nothing", "2a", 2, 4)[:2], (None, None))
        self.assertEqual(registers.parse_reg("<reg>2a zz 00ab</reg>", "2a", 2, 4)[:2], (None, "00ab"))
        self.assertEqual(registers.parse_reg("<reg>2a 3f</reg>", "2a", 2, 4)[:2], (None, None))

    def test_wrong_address_is_reported_values_still_apply(self):
        pub, _, _, errs, notes = registers.parse_reg("<reg>11 3f 00ab</reg>", "2a", 2, 4)
        self.assertEqual(pub, "3f")
        self.assertTrue(errs and "2a" in notes[0])

    def test_tags_quoted_in_notes_or_thinking_do_nothing(self):
        raw = ('<think><reg>2a aa aaaa</reg></think><reg>2a 3f 00ab</reg>'
               '<protocol>format is <reg>2a bb bbbb</reg> and <move to="00"/></protocol><log><out>ffff</out></log>')
        self.assertEqual(registers.parse_reg(raw, "2a", 2, 4)[:2], ("3f", "00ab"))
        self.assertNotIn("<move", registers.action_text(raw))
        self.assertNotIn("<out", registers.action_text(raw))


class Visibility(Base):
    def test_who_sees_what(self):
        _, meta, _ = self.world(tell_cells=False)
        bank = registers.bank_for(meta)
        bank.priv[3], bank.pub[3] = "beef", "c4"
        _, gold = registers.value_at(meta, bank, 1)
        inp = next(a["id"] for a in meta["agents"] if registers.address(a) == "00")
        out = next(a["id"] for a in meta["agents"] if registers.address(a) == "22")
        for i in range(16):
            block = registers.tick_block(meta, bank, i, 1)
            self.assertIn("c4", block)                                      # public: everyone
            self.assertEqual("beef" in block, i == 3)                       # private: its owner
            self.assertEqual(gold in block, i == inp)                       # input: its position
            self.assertEqual("output register" in block, i == out)
        self.assertNotIn(gold, sim.system_prompt(meta, meta["agents"][0]["seed"]))
        self.assertNotIn("is at position", sim.system_prompt(meta, meta["agents"][0]["seed"]))

    def test_size_never_shown(self):
        cfg = config()
        cfg["llm"]["pools"] = [{"name": "zz08b", "count": 8}, {"name": "zz4b", "count": 6}, {"name": "zz27b", "count": 2}]
        sim.set_width(cfg)
        _, meta = sim.new_run(cfg, "pools")
        states = sim.restore_states(meta, [])
        self.assertEqual(sorted(s.pool for s in states).count("zz08b"), 8)
        rec = asyncio.run(sim.step(1, meta, states, registers.FakeRegisterAgent(cfg["regfake"])))
        for a in rec["agents"]:
            self.assertIn("zz", a["pool"])
            self.assertNotIn("zz", a["prompt"])
        self.assertNotIn("zz", sim.system_prompt(meta, meta["agents"][0]["seed"]))

    def test_the_exact_format_shows_the_led_line_first(self):
        _, meta, _ = self.world()
        text = sim.system_prompt(meta, meta["agents"][0]["seed"])
        tail = text.split("in exactly this format:\n", 1)[1]
        self.assertTrue(tail.startswith("<reg>aa pp ssss</reg>\n<log>"))

    def test_old_action_text_replaced(self):
        cfg = config(actions=True, moves=True)
        text = sim.rules_text(cfg)
        self.assertNotIn("<say", text)
        self.assertIn("<remember", text)
        self.assertIn("<move", text)


class Scoring(Base):
    def out_node(self, meta):
        return next(a["id"] for a in meta["agents"] if registers.address(a) == "22")

    def test_exact_current_value_only(self):
        _, meta, _ = self.world(input_period=4)
        bank = registers.bank_for(meta)
        i = self.out_node(meta)
        _, v0 = registers.value_at(meta, bank, 1)
        raws = [self.line(meta, j) for j in range(16)]
        raws[i] = self.line(meta, i, extra=f"<out>{v0}</out>")
        snap, _ = registers.apply_tick(meta, bank, 2, raws)
        self.assertEqual((snap["delivered"], snap["events"][0]["latency"]), (1, 1))
        raws[i] = self.line(meta, i)                                         # the old value sits there: no new credit
        for t in range(3, 9):
            snap, _ = registers.apply_tick(meta, bank, t, raws)
            self.assertEqual(snap["delivered"], 1)

    def test_only_the_output_position_writes(self):
        _, meta, _ = self.world()
        bank = registers.bank_for(meta)
        _, v0 = registers.value_at(meta, bank, 1)
        i = (self.out_node(meta) + 1) % 16
        raws = [self.line(meta, j) for j in range(16)]
        raws[i] = self.line(meta, i, extra=f"<out>{v0}</out>")
        snap, res = registers.apply_tick(meta, bank, 1, raws)
        self.assertEqual((snap["delivered"], snap["out"]), (0, "0000"))
        self.assertIn("no output register", res[i]["notices"][0])

    def test_input_stream_is_seeded_and_never_stale(self):
        _, meta, _ = self.world(input_period=1)
        bank = registers.bank_for(meta)
        vals = [registers.value_at(meta, bank, t)[1] for t in range(1, 300)]
        self.assertTrue(all(a != b for a, b in zip(vals, vals[1:])))
        self.assertTrue(all(v.strip("0") for v in vals))
        self.assertEqual(vals, [registers.value_at(meta, bank, t)[1] for t in range(1, 300)])

    def test_a_dropped_turn_writes_nothing(self):
        _, meta, _ = self.world()
        bank = registers.bank_for(meta)
        bank.pub[0] = "ab"
        raws = [None] + [self.line(meta, j) for j in range(1, 16)]
        snap, res = registers.apply_tick(meta, bank, 1, raws)
        self.assertEqual(snap["pub"][0], "ab")
        self.assertEqual(res[0]["errors"], [])


class Moves(Base):
    def setUp(self):
        super().setUp()
        _, self.meta, _ = self.world(moves=True, move_cooldown=5)
        self.bank = registers.bank_for(self.meta)
        self.a = 0
        self.b = sorted(registers.neighbours(self.meta["agents"][0]))[0]
        self.A = registers.address(self.meta["agents"][self.a])
        self.B = registers.address(self.meta["agents"][self.b])

    def silent(self):
        return [self.line(self.meta, j) for j in range(16)]

    def propose(self, t=1):
        raws = self.silent()
        raws[self.a] = self.line(self.meta, self.a, extra=f'<move to="{self.B}"/>')
        return registers.apply_tick(self.meta, self.bank, t, raws)

    def others(self):
        ag = self.meta["agents"]
        return (registers.neighbours(ag[self.a]) | registers.neighbours(ag[self.b])) - {self.a, self.b}

    def test_nothing_moves_on_the_proposal_tick(self):
        self.propose()
        self.assertEqual(registers.address(self.meta["agents"][self.a]), self.A)
        self.assertEqual(len(self.bank.proposals), 1)

    def test_target_and_neighbours_are_asked_others_not(self):
        self.propose()
        others = self.others()
        for i in range(16):
            block = registers.tick_block(self.meta, self.bank, i, 2)
            self.assertEqual("asks to swap with you" in block, i == self.b)
            self.assertEqual(f"asks to swap with {self.B}" in block, i in others)

    def test_yes_with_silent_neighbours_swaps(self):
        self.propose()
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertEqual([e["kind"] for e in snap["events"]], ["swap"])
        self.assertEqual(registers.address(self.meta["agents"][self.a]), self.B)
        self.assertIn(self.B, " ".join(self.bank.news[self.a]))

    def test_silent_target_expires(self):
        self.propose()
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, self.silent())
        self.assertEqual(snap["events"][0]["why"], "the other process did not answer yes")
        self.assertEqual(registers.address(self.meta["agents"][self.a]), self.A)
        self.assertEqual(self.bank.proposals, [])

    def test_neighbour_veto(self):
        self.propose()
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        n = sorted(self.others())[0]
        raws[n] = self.line(self.meta, n, extra=f'<no from="{self.A}"/>')
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertEqual(snap["events"][0]["why"], "a neighbour said no")

    def test_dropped_neighbour_is_not_a_veto(self):
        self.propose()
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        raws[sorted(self.others())[0]] = None
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertEqual(snap["events"][0]["kind"], "swap")

    def test_majority(self):
        self.meta["config"]["registers"]["consent"] = "majority"
        self.propose()
        others = sorted(self.others())
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        for n in others[:len(others) // 2]:                                   # exactly half is not a majority
            raws[n] = self.line(self.meta, n, extra=f'<yes from="{self.A}"/>')
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertEqual(snap["events"][0]["why"], "not enough neighbours said yes")
        self.propose(3)
        n = others[len(others) // 2]
        raws[n] = self.line(self.meta, n, extra=f'<yes from="{self.A}"/>')
        snap, _ = registers.apply_tick(self.meta, self.bank, 4, raws)
        self.assertEqual(snap["events"][0]["kind"], "swap")

    def test_two_swaps_on_one_node_both_cancel(self):
        ag = self.meta["agents"]
        c = sorted(registers.neighbours(ag[self.b]) - {self.a})[0]
        raws = self.silent()
        raws[self.a] = self.line(self.meta, self.a, extra=f'<move to="{self.B}"/>')
        raws[c] = self.line(self.meta, c, extra=f'<move to="{self.B}"/>')
        registers.apply_tick(self.meta, self.bank, 1, raws)
        C = registers.address(ag[c])
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/><yes from="{C}"/>')
        snap, _ = registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertEqual([e["why"] for e in snap["events"]], ["conflict", "conflict"])
        self.assertEqual(registers.address(ag[self.a]), self.A)

    def test_cannot_propose_far_away_or_during_cooldown(self):
        far = next(a for a in self.meta["agents"] if a["id"] != self.a and a["id"] not in registers.neighbours(self.meta["agents"][self.a]))
        raws = self.silent()
        raws[self.a] = self.line(self.meta, self.a, extra=f'<move to="{registers.address(far)}"/>')
        _, res = registers.apply_tick(self.meta, self.bank, 1, raws)
        self.assertIn("not next to you", res[self.a]["notices"][0])
        self.assertEqual(self.bank.proposals, [])
        self.propose(2)
        raws = self.silent()
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        registers.apply_tick(self.meta, self.bank, 3, raws)                   # swapped at t3, blocked t4..t8
        now = sorted(registers.neighbours(self.meta["agents"][self.a]))[0]
        addr = registers.address(self.meta["agents"][now])
        raws = self.silent()
        raws[self.a] = self.line(self.meta, self.a, extra=f'<move to="{addr}"/>')
        _, res = registers.apply_tick(self.meta, self.bank, 8, raws)
        self.assertIn("cannot swap again", res[self.a]["notices"][0])
        registers.apply_tick(self.meta, self.bank, 9, raws)
        self.assertEqual(len(self.bank.proposals), 1)

    def test_io_stays_with_the_position_registers_stay_with_the_node(self):
        ag = self.meta["agents"]
        self.assertEqual(self.A, "00")                                        # node 0 starts on the input cell
        self.bank.pub[self.a], self.bank.priv[self.a] = "ab", "cdef"
        self.propose()
        raws = self.silent()
        raws[self.a] = self.line(self.meta, self.a, "ab", "cdef")
        raws[self.b] = self.line(self.meta, self.b, extra=f'<yes from="{self.A}"/>')
        registers.apply_tick(self.meta, self.bank, 2, raws)
        self.assertNotIn("input register", registers.tick_block(self.meta, self.bank, self.a, 3))
        self.assertIn("input register", registers.tick_block(self.meta, self.bank, self.b, 3))
        self.assertEqual((self.bank.pub[self.a], self.bank.priv[self.a]), ("ab", "cdef"))
        self.assertEqual(registers.address(ag[self.a]), self.B)


class Runs(Base):
    def test_fake_network_delivers_every_value_it_has_time_for(self):
        path, meta, states = self.world(input_period=6)
        ticks = self.go(meta, states, 30)
        s = registers.summary(path)
        self.assertEqual((s["values_shown"], s["values_delivered"]), (5, 5))
        self.assertEqual(ticks[-1]["reg"]["delivered"], 5)

    def test_silent_network_delivers_nothing(self):
        cfg = config()
        cfg["agent"] = "echo"
        cfg["echo"]["malformed_rate"] = 0
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, "echo")
        self.go(meta, sim.restore_states(meta, []), 30)
        s = registers.summary(path)
        self.assertEqual(s["values_delivered"], 0)
        self.assertEqual(s["led_line"]["all"]["led_ok"], 0)

    def test_noise_costs_deliveries(self):
        cfg = config(input_period=6)
        cfg["regfake"]["noise"] = {"": 1.0}
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, "noisy")
        self.go(meta, sim.restore_states(meta, []), 30)
        self.assertEqual(registers.summary(path)["values_delivered"], 0)

    def test_network_keeps_working_while_nodes_move(self):
        cfg = config(input_period=6, moves=True, move_cooldown=2)
        cfg["regfake"]["move_rate"] = 0.1
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, "moving")
        ticks = self.go(meta, sim.restore_states(meta, []), 60)
        s = registers.summary(path)
        self.assertGreater(s["swaps"], 3)
        self.assertGreater(s["values_delivered"], 3)
        for r in ticks:                                                        # addresses stay a permutation
            self.assertEqual(len(set(r["reg"]["addr"])), 16)

    def test_resume_matches_a_straight_run(self):
        def run_to(rid, stops):
            cfg = config(input_period=6, moves=True, move_cooldown=2)
            cfg["regfake"]["move_rate"] = 0.1
            sim.set_width(cfg)
            path, meta = sim.new_run(cfg, rid)
            states, start = sim.restore_states(meta, []), 1
            for end in stops:
                asyncio.run(sim.run(path, meta, states, start, end))
                meta, ticks = sim.load_run(path)
                states = sim.restore_states(meta, ticks)
                start = ticks[-1]["t"] + 1
            return ticks
        straight, resumed = run_to("straight", [40]), run_to("resumed", [13, 27, 40])
        for a, b in zip(straight, resumed):
            self.assertEqual(a["reg"], b["reg"])
            self.assertEqual([x["prompt"] for x in a["agents"]], [x["prompt"] for x in b["agents"]])

    def test_ratchet_keeps_a_working_run_and_stops_a_dead_one(self):
        path, meta, states = self.world(input_period=6)
        self.assertEqual(self.go(meta, states, 30, ratchet=10)[-1]["t"], 30)
        cfg = config()
        cfg["agent"] = "echo"
        cfg["echo"]["beacon_period"] = 1000000
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, "dead")
        self.assertEqual(self.go(meta, sim.restore_states(meta, []), 30, ratchet=10)[-1]["t"], 10)


class Config(Base):
    def test_refuses_a_bus_as_wide_as_the_input(self):
        with self.assertRaises(SystemExit):
            registers.validate(config(public_hex=4, input_hex=4))

    def test_refuses_cells_outside_the_array(self):
        with self.assertRaises(SystemExit):
            registers.validate(config(output_cell="44"))
        with self.assertRaises(SystemExit):
            registers.validate(config(output_cell="00"))

    def test_programs_need_arrays(self):
        with self.assertRaises(SystemExit):
            registers.validate(config(programs=True))

    def test_pool_counts_must_fill_the_array(self):
        cfg = config()
        cfg["llm"]["pools"] = [{"name": "a", "count": 3}]
        sim.set_width(cfg)
        with self.assertRaises(SystemExit):
            sim.new_run(cfg, "short")

    def test_pools_leave_the_wiring_alone(self):
        _, plain, _ = self.world()
        cfg = config()
        cfg["llm"]["pools"] = [{"name": "a", "count": 16}]
        sim.set_width(cfg)
        _, mixed = sim.new_run(cfg, "same")
        self.assertEqual([(a["pos"], a["links"], a["seed"]) for a in plain["agents"]],
                         [(a["pos"], a["links"], a["seed"]) for a in mixed["agents"]])


def two_arrays(**over):
    cfg = config(rows=4, cols=8, topology="grid", modules=[4, 4], input_cell="00", output_cell="07", port_bits=16, input_period=12,
                 probation=12, programs=True, tell_cells=False, gate="analog")
    cfg.update({k: v for k, v in over.items() if k not in registers.DEFAULTS})
    cfg["registers"].update({k: v for k, v in over.items() if k in registers.DEFAULTS})
    cfg["llm"]["pools"] = [{"name": "small", "count": 28, "role": "gluon"}, {"name": "big", "count": 4}]
    return cfg


class Arrays(Base):
    def build(self, rid, **over):
        cfg = two_arrays(**over)
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, rid)
        at = {registers.address(a): a for a in meta["agents"]}
        for a in meta["agents"]:                     # place the big nodes by hand: two on the seam in each array
            big = registers.address(a) in ("13", "23", "14", "24")
            a["pool"] = "big" if big else "small"
            a["role"] = "" if big else "gluon"
        return path, meta, at

    def route_script(self, meta, at):
        """a seam node writes each hop of the row-0 route in its own array, one at a time."""
        script, chain = {}, [f"0{c:x}" for c in range(8)]
        def port_to(a, b):
            return next(i for i, l in enumerate(a["links"]) if l and l["agent"] == b["id"])
        t = {"a": 1, "b": 1}
        for n, cell in enumerate(chain):
            a = at[cell]
            src = "input" if n == 0 else f"p{port_to(a, at[chain[n - 1]])}"
            dst = "out" if n == 7 else f"q{port_to(a, at[chain[n + 1]])}"
            side = "a" if n < 4 else "b"
            by = at["13"] if side == "a" else at["14"]
            script[f"{by['id']}:{t[side]}"] = f'<program to="{cell}">copy {src} to {dst}</program>'
            t[side] += 13
        return script

    def route_run(self, rid, ticks=110, **over):
        path, meta, at = self.build(rid, **over)
        meta["config"]["regfake"]["script"] = self.route_script(meta, at)
        return path, self.go(meta, sim.restore_states(meta, []), ticks, path=path)

    def test_boards_stop_at_the_array_and_seam_is_told(self):
        _, meta, at = self.build("boards")
        bank = registers.bank_for(meta)
        block = registers.tick_block(meta, bank, at["13"]["id"], 1)
        self.assertIn("03:", block)
        self.assertNotIn("04:", block)
        self.assertIn("you are on a seam", block)
        self.assertNotIn("you are on a seam", registers.tick_block(meta, bank, at["11"]["id"], 1))

    def test_only_seam_nodes_program_and_only_their_own_array_and_only_gluons(self):
        _, meta, at = self.build("rights")
        bank = registers.bank_for(meta)
        def attempt(by, to):
            raws = ["" for _ in range(32)]
            raws[at[by]["id"]] = f'<program to="{to}">copy p0 to q1</program>'
            return registers.apply_tick(meta, bank, 1, raws, [["00"] * 4 for _ in range(32)])[0]["events"]
        self.assertIn("only a process on a seam", [e for e in attempt("11", "01") if e["kind"] == "program_refused"][0]["why"])
        self.assertIn("not in your array", [e for e in attempt("13", "04") if e["kind"] == "program_refused"][0]["why"])
        self.assertIn("does not accept", [e for e in attempt("13", "23") if e["kind"] == "program_refused"][0]["why"])
        self.assertEqual([e["kind"] for e in attempt("13", "02")], ["program_trial"])
        self.assertIn("already on trial", [e for e in attempt("23", "01") if e["kind"] == "program_refused"][0]["why"])

    def test_bad_programs_are_refused(self):
        rc = two_arrays()["registers"]
        for bad in ("copy p9 to q1", "copy q0 to q1", "copy p0 to p1", "copy input to private.hi", "steal p0", "", "copy p0 to q0;" * 5):
            self.assertIsNone(registers.check_program(bad, rc, 4)[0], bad)
        self.assertEqual(registers.check_program("COPY p1 TO q3; copy input to out", rc, 4)[0], "copy p1 to q3; copy input to out")

    def test_delivery_is_analog(self):
        self.assertEqual(registers.correlation("7a1f", "7a1f"), 1.0)
        self.assertEqual(registers.correlation("85e0", "7a1f"), -1.0)
        half = registers.correlation("7a00", "7aff")
        self.assertEqual(half, 0.0)
        one_bit_off = registers.correlation("7a1e", "7a1f")
        self.assertAlmostEqual(one_bit_off, 0.875)
        rng = random.Random(3)
        noise = [registers.correlation(f"{rng.getrandbits(16):04x}", f"{rng.getrandbits(16):04x}") for _ in range(4000)]
        self.assertLess(abs(sum(noise) / len(noise)), 0.02)

    def test_gate_decisions(self):
        rc = two_arrays(probation=24)["registers"]
        w = lambda signal=0.0, reach=0.0, noise=0.0, deliveries=0: dict(signal=signal, reach=reach, noise=noise, deliveries=deliveries)
        decide = lambda before, after: registers.gate_decision(rc, before, after)[0]
        self.assertTrue(decide(w(), w(signal=0.5)))                            # half a value is real progress
        self.assertFalse(decide(w(signal=0.5), w()))                           # and losing it is real damage
        self.assertTrue(decide(w(), w(reach=2.0)))                             # no signal yet, but it travelled further
        self.assertFalse(decide(w(reach=2.0), w(reach=2.0)))                   # nothing moved: undo
        self.assertFalse(decide(w(signal=0.1, reach=2.0), w(signal=0.0, reach=1.0)))
        self.assertFalse(decide(w(), w(signal=0.9, noise=0.4)))                # the route was bought with noise
        self.assertTrue(decide(w(noise=0.3), w(signal=0.9, noise=0.4)))        # the same signal, within the noise limit
        self.assertFalse(decide(w(), w(signal=0.05)))                          # inside the margin is not a gain
        rc["gate"] = "exact"
        self.assertFalse(decide(w(), w(signal=0.9)))
        self.assertTrue(decide(w(), w(deliveries=1)))

    def test_analog_gate_builds_a_route_and_delivers(self):
        path, ticks = self.route_run("analog")
        s = registers.summary(path)
        self.assertGreaterEqual(s["values_delivered"], 3)
        self.assertEqual(s["programs"]["kept"], 8)
        self.assertEqual(s["programs"]["reverted"], 0)
        self.assertGreater(s["mean_signal_last_quarter"], 0.25)   # right only after the route's 8 ticks of latency, of 12

    def test_pinned_cells_and_wiring_view(self):
        cfg = two_arrays(show_wiring=True)
        cfg["llm"]["pools"] = [{"name": "small", "count": 28, "role": "gluon"}, {"name": "big", "count": 4, "cells": ["13", "23", "14", "24"]}]
        sim.set_width(cfg)
        _, meta = sim.new_run(cfg, "pinned")
        at = {registers.address(a): a for a in meta["agents"]}
        self.assertEqual({c for c, a in at.items() if a["pool"] == "big"}, {"13", "23", "14", "24"})
        self.assertEqual(sum(a["pool"] == "small" for a in meta["agents"]), 28)
        self.assertTrue(all(registers.is_gluon(a) for a in meta["agents"] if a["pool"] == "small"))
        bank = registers.bank_for(meta)
        block = registers.tick_block(meta, bank, at["13"]["id"], 1)
        self.assertIn("wiring of your array", block)
        self.assertIn("03: ", block)
        self.assertNotIn("\n05: ", block)
        self.assertNotIn("wiring", registers.tick_block(meta, bank, at["11"]["id"], 1))

    def test_seam_nodes_see_instructions_and_trials_and_a_kept_program_counts_as_interesting(self):
        _, meta, at = self.build("view")
        bank = registers.bank_for(meta)
        by = at["13"]["id"]
        self.assertIn("instructions in your array: none yet", registers.tick_block(meta, bank, by, 1))
        raws = ["" for _ in range(32)]
        raws[by] = '<program to="02">copy p0 to q1</program>'
        registers.apply_tick(meta, bank, 1, raws, [["0000"] * 4 for _ in range(32)])
        block = registers.tick_block(meta, bank, by, 2)
        self.assertIn("02: copy p0 to q1", block)
        self.assertIn("on trial: the instruction for 02, until tick 13", block)
        self.assertNotIn("on trial", registers.tick_block(meta, bank, at["14"]["id"], 2))   # the other array's trial is not its business
        tick = {"t": 2, "agents": [], "reg": {"pub": ["00"] * 32, "delivered": 0, "events": [{"kind": "program_kept"}]}}
        self.assertTrue(registers.interesting([tick], [], 32)[0])
        tick["reg"]["events"] = [{"kind": "program_reverted"}]
        self.assertFalse(registers.interesting([tick], [], 32)[0])

    def test_exact_gate_alone_cannot_build_a_route(self):
        path, ticks = self.route_run("exact", gate="exact")
        s = registers.summary(path)
        self.assertEqual(s["values_delivered"], 0)
        self.assertEqual(s["programs"]["kept"], 0)
        self.assertEqual(s["programs"]["reverted"], 8)

    def test_a_useless_program_is_undone(self):
        path, meta, at = self.build("useless")
        meta["config"]["regfake"]["script"] = {f"{at['13']['id']}:1": '<program to="22">copy p0 to private</program>'}
        ticks = self.go(meta, sim.restore_states(meta, []), 16, path=path)
        kinds = [e["kind"] for r in ticks for e in r["reg"]["events"]]
        self.assertIn("program_reverted", kinds)
        self.assertEqual(ticks[-1]["reg"]["programs"][at["22"]["id"]], "keep everything as it is.")

    def test_gluon_gets_short_prompt_and_grammar(self):
        _, meta, at = self.build("short")
        states = sim.restore_states(meta, [])
        seen = {}
        class Spy(registers.FakeRegisterAgent):
            async def act(self, turn):
                seen[turn.state.id] = turn
                return await super().act(turn)
        asyncio.run(sim.step(1, meta, states, Spy({})))
        g, b = seen[at["22"]["id"]], seen[at["13"]["id"]]
        self.assertTrue(g.system.startswith("you pass values along"))
        self.assertLess(len(g.prompt), 200)
        self.assertIn("<reg>22 ", g.extra_body["regex"])
        self.assertNotIn("<out>", g.extra_body["regex"])
        self.assertNotIn("<out>", seen[at["07"]["id"]].extra_body["regex"])     # the output's tag only when its instruction writes it
        self.assertIn("<out>", registers.gluon_grammar(meta, at["07"]["id"], 4, "copy p3 to out")["regex"])
        self.assertNotIn(")?", registers.gluon_grammar(meta, at["07"]["id"], 4, "copy p3 to out")["regex"])
        self.assertEqual(b.extra_body, {})
        self.assertNotIn("you pass values", b.system)

    def test_moves_redraw_seams_and_seam_rights_follow_the_position(self):
        _, meta, at = self.build("moved", moves=True)
        a, b = at["13"], at["12"]
        self.assertTrue(registers.on_seam(a))
        self.assertFalse(registers.on_seam(b))
        a["pos"], b["pos"] = b["pos"], a["pos"]
        registers._rewire_meta(meta)
        self.assertFalse(registers.on_seam(a))
        self.assertTrue(registers.on_seam(b))

    def test_resume_with_programs_matches_a_straight_run(self):
        def go(rid, stops):
            path, meta, at = self.build(rid)
            meta["config"]["regfake"]["script"] = self.route_script(meta, at)
            states, start = sim.restore_states(meta, []), 1
            for end in stops:
                asyncio.run(sim.run(path, meta, states, start, end))
                # the file holds the config from before the script was added: restore it by hand
                keep = meta["config"]["regfake"]["script"]
                roles = [(a.get("pool"), a.get("role")) for a in meta["agents"]]
                meta, ticks = sim.load_run(path)
                meta["config"]["regfake"]["script"] = keep
                for a, (p, r) in zip(meta["agents"], roles):
                    a["pool"], a["role"] = p, r
                states = sim.restore_states(meta, ticks)
                start = ticks[-1]["t"] + 1
            return ticks
        for a, b in zip(go("str2", [30]), go("res2", [7, 19, 30])):
            self.assertEqual(a["reg"], b["reg"])
            self.assertEqual([x["prompt"] for x in a["agents"]], [x["prompt"] for x in b["agents"]])


class UnitGate(Base):
    def build(self, rid, **over):
        cfg = two_arrays(gate="unit", **over)
        sim.set_width(cfg)
        path, meta = sim.new_run(cfg, rid)
        at = {registers.address(a): a for a in meta["agents"]}
        for a in meta["agents"]:
            big = registers.address(a) in ("13", "23", "14", "24")
            a["pool"] = "big" if big else "small"
            a["role"] = "" if big else "gluon"
        return path, meta, at

    def route(self, meta, at, reverse=True):
        """every hop of the row-0 route, one tick apart, from the output end back to the input."""
        chain = [f"0{c:x}" for c in range(8)]
        def port_to(a, b):
            return next(i for i, l in enumerate(a["links"]) if l and l["agent"] == b["id"])
        script, tick = {}, 1
        for n in (reversed(range(8)) if reverse else range(8)):
            a = at[chain[n]]
            src = "input" if n == 0 else f"p{port_to(a, at[chain[n - 1]])}"
            dst = "out" if n == 7 else f"q{port_to(a, at[chain[n + 1]])}"
            by = at["13"] if n < 4 else at["14"]
            script[f"{by['id']}:{tick}"] = f'<program to="{chain[n]}">copy {src} to {dst}</program>'
            tick += 1
        return script

    def run_route(self, rid, ticks=60, noise=None, reverse=True):
        path, meta, at = self.build(rid)
        meta["config"]["regfake"]["script"] = self.route(meta, at, reverse)
        if noise:
            meta["config"]["regfake"]["noise"] = noise
        rows = self.go(meta, sim.restore_states(meta, []), ticks, path=path)
        return path, rows, registers.summary(path)

    def test_a_route_built_backwards_still_works(self):
        _, rows, s = self.run_route("back")
        self.assertEqual(s["programs"]["kept"], 8)
        self.assertEqual(s["programs"]["failed_test"], 0)
        self.assertGreaterEqual(s["values_delivered"], 4)
        kept = [r["t"] for r in rows for e in r["reg"]["events"] if e["kind"] == "program_kept"]
        self.assertEqual(kept, list(range(1, 9)))                              # no waiting between hops

    def test_a_gluon_that_garbles_fails_its_test(self):
        _, rows, s = self.run_route("garble", ticks=12, noise={"small": 1.0})
        self.assertEqual(s["programs"]["kept"], 0)
        self.assertEqual(s["programs"]["failed_test"], 8)
        self.assertTrue(all(p.startswith("keep") for p in rows[-1]["reg"]["programs"]))
        news = " ".join(x for r in rows for n in r["reg"]["news"] for x in n)
        self.assertIn("not installed", news)

    def test_test_values_are_not_live_values_and_go_only_to_the_target(self):
        path, meta, at = self.build("side")
        seen = []
        class Spy(registers.FakeRegisterAgent):
            async def act(self, turn):
                seen.append(turn)
                return await super().act(turn)
        meta["config"]["regfake"]["script"] = {f"{at['13']['id']}:1": f'<program to="00">copy input to q0</program>'}
        states = sim.restore_states(meta, [])
        live = registers.value_at(meta, registers.bank_for(meta), 1)[1]
        rec = asyncio.run(sim.step(1, meta, states, Spy(meta["config"]["regfake"])))
        tests = [t for t in seen if t.prompt.startswith("instruction: copy input to q0")]
        self.assertEqual(len(tests), 6)
        self.assertEqual({t.state.id for t in tests}, {at["00"]["id"]})
        self.assertNotIn(live, " ".join(t.prompt for t in tests))
        self.assertEqual(rec["reg"]["programs"][at["00"]["id"]], "copy input to q0")
        self.assertEqual(len(rec["agents"]), 32)                               # test calls never appear as agents' turns

    def test_position_rules_and_only_one_test_per_target(self):
        _, meta, at = self.build("rules")
        bank = registers.bank_for(meta)
        raws = ["" for _ in range(32)]
        raws[at["13"]["id"]] = '<program to="02">copy input to q0</program>'
        raws[at["14"]["id"]] = '<program to="05">copy p0 to out</program>'
        snap, _ = registers.apply_tick(meta, bank, 1, raws, [["0000"] * 4 for _ in range(32)])
        whys = " ".join(e["why"] for e in snap["events"] if e["kind"] == "program_refused")
        self.assertIn("02 has no input register", whys)
        self.assertIn("05 has no output register", whys)
        self.assertEqual(bank.pending, [])
        raws = ["" for _ in range(32)]
        raws[at["13"]["id"]] = '<program to="02">copy p0 to q0</program>'
        raws[at["23"]["id"]] = '<program to="02">copy p1 to q0</program>'
        snap, _ = registers.apply_tick(meta, bank, 2, raws, [["0000"] * 4 for _ in range(32)])
        self.assertEqual(len(bank.pending), 1)
        self.assertIn("being tested", " ".join(e["why"] for e in snap["events"] if e["kind"] == "program_refused"))

    def test_expected_outputs_and_scoring(self):
        rc = two_arrays(gate="unit")["registers"]
        have = {"p0": "1111", "p1": "2222", "public": "ab", "private": "cdef", "input": "7a1f"}
        self.assertEqual(registers.expected_outputs("copy input to q1", have), {"q1": "7a1f"})
        self.assertEqual(registers.expected_outputs("copy input.hi to public; copy input.lo to private.hi", have), {"public": "7a", "private.hi": "1f"})
        self.assertEqual(registers.expected_outputs("copy p0 to private; copy private to q1", have), {"private": "1111", "q1": "1111"})
        good = "<reg>00 ab cdef</reg>\n<log>ok</log>\n<q0>0000</q0>\n<q1>7a1f</q1>\n<q2>0000</q2>\n<q3>0000</q3>\n"
        self.assertEqual(registers.score_unit_reply(good, {"q1": "7a1f"}, "00", rc, 4), 1.0)
        self.assertEqual(registers.score_unit_reply(good.replace("7a1f", "85e0"), {"q1": "7a1f"}, "00", rc, 4), -1.0)
        self.assertEqual(registers.score_unit_reply("nonsense", {"q1": "7a1f"}, "00", rc, 4), 0.0)
        half = registers.score_unit_reply(good, {"q1": "7a1f", "q0": "1111"}, "00", rc, 4)
        self.assertLess(half, 0.8)                                             # one of two destinations right does not pass

    def test_a_malformed_program_is_refused_not_a_crash(self):
        _, meta, at = self.build("bad-syntax")
        bank = registers.bank_for(meta)
        for bad in ("steal p0", "copy input to", "copy p9 to q0", ""):
            raws = ["" for _ in range(32)]
            raws[at["13"]["id"]] = f'<program to="02">{bad}</program>'
            snap, res = registers.apply_tick(meta, bank, 1, raws, [["0000"] * 4 for _ in range(32)])
            self.assertEqual([e["kind"] for e in snap["events"]], ["program_refused"], bad)
            self.assertTrue(any("<program> was ignored" in n for n in res[at["13"]["id"]]["notices"]))

    def test_several_programs_in_one_reply_are_all_read(self):
        _, meta, at = self.build("several")
        bank = registers.bank_for(meta)
        raws = ["" for _ in range(32)]
        raws[at["13"]["id"]] = "".join(f'<program to="{c}">copy p0 to q1</program>' for c in ("01", "02", "03"))
        registers.apply_tick(meta, bank, 1, raws, [["0000"] * 4 for _ in range(32)])
        self.assertEqual(len(bank.pending), 3)

    def test_clobbering_other_outputs_fails_and_half_writes_score_only_their_half(self):
        rc = two_arrays(gate="unit")["registers"]
        have = {"p0": "7a1f", "p1": "2222", "p2": "3333", "p3": "4444", "public": "ab", "private": "cdef", "input": "1234"}
        want = registers.expected_outputs("copy p0 to q1", have)
        clean = "<reg>00 ab cdef</reg>\n<log>ok</log>\n<q0>0000</q0>\n<q1>7a1f</q1>\n<q2>0000</q2>\n<q3>0000</q3>\n"
        self.assertEqual(registers.score_unit_reply(clean, want, "00", rc, 4, have), 1.0)
        messy = clean.replace("<q0>0000", "<q0>9999").replace("<q2>0000", "<q2>8888").replace("ab cdef", "01 0000")
        self.assertLess(registers.score_unit_reply(messy, want, "00", rc, 4, have), 0.8)
        half = registers.expected_outputs("copy input.hi to private.hi", have)
        wrong = clean.replace("ab cdef", "ab eeef")                                # 'ee' for '12', low half untouched
        self.assertLess(registers.score_unit_reply(wrong, half, "00", rc, 4, have), 0.5)

    def test_one_clobbered_output_fails_and_hi_lo_are_defined(self):
        rc = two_arrays(gate="unit")["registers"]
        have = {"p0": "7a1f", "p1": "2222", "p2": "3333", "p3": "4444", "public": "ab", "private": "cdef", "input": "1234"}
        want = registers.expected_outputs("copy p0 to q0", have)
        spray = "<reg>00 ab cdef</reg>\n<log>ok</log>\n<q0>7a1f</q0>\n<q1>7a1f</q1>\n<q2>0000</q2>\n<q3>0000</q3>\n"
        self.assertLess(registers.score_unit_reply(spray, want, "00", rc, 4, have), 0.8)
        self.assertIn(".hi means", registers.gluon_system(two_arrays(gate="unit")))

    def test_seam_nodes_see_where_the_value_got_to(self):
        _, meta, at = self.build("reach-view", show_wiring=True)
        bank = registers.bank_for(meta)
        self.assertNotIn("has reached", registers.tick_block(meta, bank, at["13"]["id"], 1))
        k, gold = registers.value_at(meta, bank, 1)
        outs = [["0000"] * 4 for _ in range(32)]
        outs[at["01"]["id"]][0] = gold
        outs[at["05"]["id"]][0] = gold
        registers.apply_tick(meta, bank, 1, ["" for _ in range(32)], outs)
        block = registers.tick_block(meta, bank, at["13"]["id"], 2)
        self.assertIn("reached, in your array: 01", block)
        self.assertNotIn("05", block.split("reached, in your array:")[1].split("\n")[0])

    def test_breaking_a_working_route_is_undone(self):
        _, meta, at = self.build("broken-route", probation=4, input_period=4)
        bank = registers.bank_for(meta)
        i = at["02"]["id"]
        self._watch(meta, bank, i, at["13"]["id"], before_signal=0.2)
        bank.watches[0]["before"]["deliveries"] = 1                         # the signal fall alone is inside the margin
        for t in range(2, 6):
            snap, _ = registers.apply_tick(meta, bank, t, ["" for _ in range(32)], [["0000"] * 4 for _ in range(32)])
        self.assertEqual([e["why"] for e in snap["events"] if e["kind"] == "program_reverted"], ["fewer values arrived after it"])

    def test_tests_for_several_instructions_all_run_at_once(self):
        path, meta, at = self.build("together")
        calls = {"now": 0, "peak": 0}
        class Slow(registers.FakeRegisterAgent):
            async def act(self, turn):
                if turn.prompt.startswith("instruction: copy"):
                    calls["now"] += 1
                    calls["peak"] = max(calls["peak"], calls["now"])
                    await asyncio.sleep(0)
                    calls["now"] -= 1
                return await super().act(turn)
        meta["config"]["regfake"]["script"] = {f"{at['13']['id']}:1": "".join(f'<program to="{c}">copy p0 to q1</program>' for c in ("01", "02", "03"))}
        states = sim.restore_states(meta, [])
        rec = asyncio.run(sim.step(1, meta, states, Slow(meta["config"]["regfake"])))
        self.assertEqual(sum(e["kind"] == "program_kept" for e in rec["reg"]["events"]), 3)
        self.assertGreater(calls["peak"], 6)

    def _watch(self, meta, bank, i, by, before_signal=0.0):
        bank.programs[i] = "copy p0 to q0"
        bank.watches.append({"target": i, "by": by, "old": "keep everything as it is.", "new": "copy p0 to q0", "t0": 1,
                             "before": {"signal": before_signal, "deliveries": 0, "reach": 0, "noise": 0.0}})

    def test_one_churning_node_is_undone(self):
        _, meta, at = self.build("noise-undo", probation=4)
        bank = registers.bank_for(meta)
        i = at["02"]["id"]
        self._watch(meta, bank, i, at["13"]["id"])
        rng = random.Random(1)
        for t in range(2, 6):
            outs = [["0000"] * 4 for _ in range(32)]
            outs[i] = [f"{rng.getrandbits(16):04x}" for _ in range(4)]
            snap, _ = registers.apply_tick(meta, bank, t, ["" for _ in range(32)], outs)
        self.assertEqual(bank.programs[i], "keep everything as it is.")
        self.assertEqual([e["why"] for e in snap["events"] if e["kind"] == "program_reverted"], ["it added too much noise"])

    def test_an_instruction_that_breaks_delivery_is_undone(self):
        _, meta, at = self.build("break", probation=4, input_period=4)
        bank = registers.bank_for(meta)
        i = at["02"]["id"]
        self._watch(meta, bank, i, at["13"]["id"], before_signal=1.0)     # the output was right before; nothing arrives after
        for t in range(2, 6):
            snap, _ = registers.apply_tick(meta, bank, t, ["" for _ in range(32)], [["0000"] * 4 for _ in range(32)])
        self.assertEqual(bank.programs[i], "keep everything as it is.")
        self.assertEqual([e["why"] for e in snap["events"] if e["kind"] == "program_reverted"], ["less of the values arrived after it"])

    def test_a_quiet_good_instruction_stays(self):
        _, meta, at = self.build("stays", probation=4)
        bank = registers.bank_for(meta)
        i = at["02"]["id"]
        self._watch(meta, bank, i, at["13"]["id"])
        for t in range(2, 6):
            registers.apply_tick(meta, bank, t, ["" for _ in range(32)], [["0000"] * 4 for _ in range(32)])
        self.assertEqual(bank.programs[i], "copy p0 to q0")

    def test_resume_with_the_unit_gate_matches_a_straight_run(self):
        def go(rid, stops):
            path, meta, at = self.build(rid)
            keep = self.route(meta, at)
            meta["config"]["regfake"]["script"] = keep
            states, start = sim.restore_states(meta, []), 1
            for end in stops:
                asyncio.run(sim.run(path, meta, states, start, end))
                roles = [(a.get("pool"), a.get("role")) for a in meta["agents"]]
                meta, ticks = sim.load_run(path)
                meta["config"]["regfake"]["script"] = keep
                for a, (p, r) in zip(meta["agents"], roles):
                    a["pool"], a["role"] = p, r
                states = sim.restore_states(meta, ticks)
                start = ticks[-1]["t"] + 1
            return ticks
        for a, b in zip(go("u1", [30]), go("u2", [5, 13, 30])):
            self.assertEqual(a["reg"], b["reg"])
            self.assertEqual([x["prompt"] for x in a["agents"]], [x["prompt"] for x in b["agents"]])


class GluonCheck(Base):
    def test_a_good_gluon_passes_every_case_and_a_garbler_none(self):
        import reg_gluon_check
        cfg = sim.load_config(str(ROOT / "config.registers-two-arrays.toml"))
        good = asyncio.run(reg_gluon_check.check(cfg, registers.FakeRegisterAgent({}), 4, "small"))
        self.assertEqual([r["program"] for r in good if not r["passes"]], [])
        bad = asyncio.run(reg_gluon_check.check(cfg, registers.FakeRegisterAgent({"noise": {"small": 1.0}}), 4, "small"))
        self.assertEqual([r["program"] for r in bad if r["passes"] and "public to public" not in r["program"]], [])


class Mixed(Base):
    def test_each_pool_reaches_its_own_model_and_endpoint(self):
        cfg = copy.deepcopy(sim.DEFAULTS["llm"])
        cfg.update(model="big", base_url="http://big/v1", temperature="", max_tokens=9, extra_body={"reasoning_effort": "low"})
        cfg["pools"] = [{"name": "small", "count": 1, "model": "lil", "base_url": ["http://lil/v1"], "max_tokens": 5,
                         "extra_body": {"reasoning_effort": "medium"}},
                        {"name": "big", "count": 1}]
        seen = []

        def handler(req):
            body = json.loads(req.content)
            seen.append((req.url.host, body["model"], body["max_tokens"], body["reasoning_effort"]))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        agent = sim.MixedLLMAgent(cfg)
        for a in agent.pools.values():
            a.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def go():
            for pool in ("small", "big"):
                st = sim.AgentState(id=0, seed="00000000", pool=pool)
                turn = sim.Turn(t=1, state=st, inputs=[], changed=[], system="s", prompt="p")
                self.assertEqual((await agent.act(turn)).raw, "ok")
            await agent.close()
        asyncio.run(go())
        self.assertEqual(seen, [("lil", "lil", 5, "medium"), ("big", "big", 9, "low")])


    def test_a_pools_extra_body_replaces_the_shared_one(self):
        cfg = {"model": "x", "extra_body": {"reasoning_effort": "medium"}, "max_tokens": 9}
        got = sim.pool_config(cfg, {"name": "a", "count": 1, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
        self.assertEqual(got["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(sim.pool_config(cfg, {"name": "b", "count": 1})["extra_body"], {"reasoning_effort": "medium"})


class CopyTest(Base):
    def test_a_perfect_and_a_garbling_model(self):
        cfg = config()
        cfg["llm"]["pools"] = [{"name": "good", "count": 8}, {"name": "bad", "count": 8}]
        out = asyncio.run(reg_copy_test.run_all(cfg, 12, lambda name: reg_copy_test.Scripted(0.0 if name == "good" else 1.0)))
        self.assertEqual(out["good"]["copy_exact"], 1.0)
        self.assertEqual(out["good"]["input_exact"], 1.0)
        self.assertEqual(out["good"]["hold_private"], 1.0)
        self.assertEqual(out["good"]["format_ok"], 1.0)
        self.assertTrue(out["good"]["passes"])
        self.assertEqual(out["bad"]["copy_exact"], 0.0)
        self.assertFalse(out["bad"]["passes"])


if __name__ == "__main__":
    unittest.main()
