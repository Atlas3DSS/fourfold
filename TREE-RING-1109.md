# Tree and ring sums instead of flooding (work item #440)

fabe-integration, started 2026-11-10 00:30Z, answering an internal brief on why flooding cannot scale. Code: `sim.py` (typed writes `as="tree"` and
`as="ring"`, exact sums like `469+417` inside typed numbers, `protocol_hint`, `tree_roots`, agent kind `scripted`),
`scripted.py`, `experiments/tree_ring/`, `tests/test_tree_ring.py`.

## Pass bars, frozen 2026-11-10 01:25Z before any LLM result on these arms was read

"Exact" is the evaluator's `<sum>` answer equal to the true total. "Wrong" is a numeric `<sum>` answer that is not the true
total; "unknown" or no number is an abstention and is counted separately, not as wrong.

| arm | world | pass bar |
|---|---|---|
| T-told (one tree) | 10x10 torus, seed 20261108 | at least 90/100 exact by tick 60, at most 2 wrong |
| T-verify (two trees, answer only if they agree) | 10x10 torus, same seed | at least 90/100 exact by tick 60, 0 wrong |
| ring sum | tesseract side 4, 256 agents, same seed | at least 230/256 exact by tick 40, at most 5 wrong |
| swarm, one tree | 4x28 torus, 7 arrays, seams every 2 ticks, 112 agents | at least 101/112 exact by tick 80, at most 2 wrong |

Baseline (flooding id:number pairs): 10x10 0/100 at tick 60; 8x8 0/64 at tick 40; tesseract stopped at tick 14.

Rules I am holding myself to:
- 4x4 runs are smoke tests for the wording of the method text and are not scored against a bar. The method text may change
  only because of a 4x4 smoke. The text the first large run uses is the text every large run uses.
- A run may be evaluated early (eval "as if it ended at tick t") to save cards. If it passes early it passes; if it does not, it
  runs on to the tick in its bar before it is called a fail.
- Help given to the agents beyond the baseline recipe, to be listed beside every result: the method text, the typed write,
  the host doing the additions, the host-designated root(s), "p_i and q_i lead to the same neighbour" (tree) or "q_d leads
  round a ring back into p_d" (ring), and the decoded view of the agent's own outputs each tick.

## Amendment, 2026-11-10 01:55Z (after the bars, before any large ring run and before any Qwen two-tree run)

Two changes, both made for reasons found outside the bars, and both stated here so nobody has to trust me on timing:

1. **Ring rule.** The scripted model with 5% of replies missing deadlocked on the tesseract (0/10 solved, never a wrong total):
   an agent that had finished a ring left old entries on its port, its neighbour kept seeing two foreign entries and never
   had room for its own. Rule 2 now says the port holds exactly the current entries and is cleared with `none` when there
   are none. After the change: 10/10 solved at 5% and at 20% missing, no wrong totals. The first Haiku ring smoke (16/16)
   used the old wording; a second smoke uses the new one. No large ring run had started.
2. **Two-tree text and a host check.** In the first Haiku two-tree 10x10 run (`haiku-10x10-tree-verify-1110`, which I had
   launched without a two-tree 4x4 smoke: my mistake) agent 63 wrote "A wave; B wave" on three outputs at once, one of
   which was its tree-B parent's. Its parent took that as "not below me" and reported without agent 63's 11 processes.
   Tree A gave 49,689 over 100 (right); tree B gave 44,294 over 89 (wrong). That run stands as run and is reported as
   such. For later two-tree runs: one more sentence in the two-tree text ("decide each output's A entry and B entry
   separately, and never put a tree's wave on that tree's parent output"), and the host now refuses a wave on the one
   output that has no entry for a tree whose wave is already on the other three (the root excepted). The one-tree text
   is unchanged.

## Results so far (2026-11-10 02:15Z)

### 1. Perfect scripted agents, through the real sim (`experiments/tree_ring/floor.py`, seed 20261108)

| world | agents | flood: all exact | one tree: first / all exact | two trees: all exact | ring: all exact |
|---|---|---|---|---|---|
| 4x4 torus | 16 | 9 | 10 / 14 | 14 | |
| 8x8 torus | 64 | 33 | 18 / 26 | 26 | |
| 10x10 torus | 100 | 51 | 22 / 32 | 32 | |
| 4x28 torus, 7 arrays, seams every 2 ticks | 112 | never (400-tick cap) | 34 / 50 | 50 | |
| 20x20 torus | 400 | not run | 42 / 62 | 62 | |
| tesseract side 3 | 81 | 40 | | | 13 |
| tesseract side 4 | 256 | 128 | | | 17 |
| tesseract side 5 | 625 | not run | | | 21 |

These match the brief's model (flood N/2; tree about 3 x diameter). Two things the model did not show:
- **Flooding never finishes across slow seams.** A seam that delivers every second tick drops every other pair a flooding
  agent writes, because the agent has moved on to the next two pairs. Tree and ring entries stay on the port, so seams
  only delay them.
- **The ring sum here takes 4 ticks a ring, not 3** (17 ticks on side 4, not 12): each agent waits for its own entry
  to come back carrying the ring's sum, which is also how it knows the ring is complete.

With replies randomly missing (10 worlds each; "solved" = every agent exact within 400 ticks; no arm ever held a wrong total):

| world | method | 5% missing | 20% missing |
|---|---|---|---|
| 10x10 | flood | 5/10 solved | 0/10 |
| 10x10 | one tree | 10/10, median tick 32.5 | 10/10, median 37.5 |
| 10x10 | two trees | 10/10, median 34 | 10/10, median 38.5 |
| 4x28 with seams | one tree | 10/10, median 52 | not run |
| tesseract side 4 | ring (amended rule) | 10/10, median 39 | 10/10, median 51.5 |

### 2. Claude Haiku 5.5 (headless claude -p, effort medium)

| run | world | result | against its bar |
|---|---|---|---|
| haiku-4x4-tree-smoke1-1110 | 4x4, one tree | 16/16 exact, all held the total at tick 17 (floor 14) | smoke |
| haiku-tess2-ring-smoke1-1110, -smoke2- | 16-agent tesseract, ring (old and amended rule) | 16/16 exact both times | smoke |
| haiku-4x4-tree-verify-smoke2-1110 | 4x4, two trees (amended) | 16/16 exact, trees agree, tick 20 | smoke |
| **haiku-10x10-tree-told-1110** | 10x10, one tree | **100/100 exact, 0 wrong**; all held the total at tick 34 (floor 32); every count answer 100; 2 rejected writes; 28 s a tick; $3.20 | **pass** |
| **haiku-4x28-arrays-tree-1110** | 4x28 swarm with seams, one tree | **112/112 exact, 0 wrong**; all held the total at tick 64 (floor 50); every count answer 112; 0 rejected; $5.62 | **pass** |
| haiku-10x10-tree-verify-1110 | 10x10, two trees, first wording | tree A 49,689 over 100 (right), tree B 44,294 over 89 (wrong, see amendment). Answers: 79 exact, 1 wrong (it answered 100), 20 no number; $6.38 | **fail** (needs 90 exact, 0 wrong) |
| **haiku-10x10-tree-verify-v2-1110** | 10x10, two trees, amended | **100/100 exact, 0 wrong, 0 no number**; first exact tick 38, all held the total at tick 48 (floor 32), stopped itself at 48; every count answer 100; no agent ever held a wrong total; 3 rejected writes; 42 s a tick; $5.41 for all 48 ticks | **pass** |
| **haiku-tesseract4-ring-1110** | tesseract side 4, ring (amended rule) | **186/256 exact, 70 wrong, 0 no number** at eval_t40; first exact tick 17, never all exact (183 held the total at tick 40); 67 agents ever held a wrong total. Main wrong answer 129,797 (56 agents, 46 of them counting 257) = true 129,219 + 578, agent 82's number counted twice; the other 14 wrong answers are 138,176, 153,329, 153,933 (4 each) and 224,428 (2), with counts 273-448. 4 rejected writes; 112 s a tick; $14.68 for all 40 ticks | **fail** |

Earlier Haiku runs on flooding, for comparison: 4x28 0/112 exact after 100 ticks, twice ($10.80 and $13.50, 8.2M and 10.5M
output tokens; the tree run used 2.9M). The last two Haiku runs were stopped and resumed from their last tick by a second agent (test), which is why their rows
were written by it.

What the failed two-tree run shows: the second tree did what it is there for. One slip in one agent removed 11 processes
from tree B without any error anywhere, and the only sign was that the two totals differed. Twenty agents then abstained
as told; 79 answered tree A's total anyway, which happened to be the right one. With one tree and the same slip, all 100
would have reported a wrong sum with full confidence.

### 3. Qwen3.8-27B at medium thinking (four Flare servers on Quincy, job fourfold-tree-ring-chain)

| run | world | result | against its bar |
|---|---|---|---|
| q-4x4-torus-sum-tree-smoke-1110 | 4x4, one tree | 16/16 exact, 0 wrong; all held the total at tick 14, which is the perfect-agent floor; 23 s a tick; median prompt 1,884 tokens at tick 1, 2,537 at tick 10 | smoke |
| **q-10x10-torus-sum-tree-told-1110** | 10x10, one tree | **100/100 exact, 0 wrong, 0 no number**; first exact total at tick 23, all held it at tick 33 (floor 32) and the run stopped itself; every count answer 100; 0 rejected writes, 0 missing replies in 3,300 turns; 202 s a tick; median prompt 1,885 tokens at tick 1, 2,500 at tick 10, 3,326 at tick 30 | **pass** |
| q-tesseract2-sum-ring-smoke-1110 | 16-agent tesseract, ring, first rule | 0/16 exact, 13 wrong, 3 no number; the 256-agent run was skipped by the gate | **fail** (smoke) |
| **q-10x10-torus-sum-tree-verify-1110** | 10x10, two trees (amended text and host check) | **0/100 exact, 6 wrong, 94 no number** at eval_t41. Both trees came out wrong and disagreed: tree A 49,688 over 100 (true 49,689, one short), tree B 47,863 over 96. 94 agents saw the disagreement and gave no number; 6 answered tree A's 49,688. All 100 held both totals at tick 41 and the run stopped itself there, so the totals were final and running to tick 60 would not change them. Every count answer 100; 0 rejected writes, 0 missing replies; 226 s a tick; median prompt 2,028 tokens at tick 1, 2,843 at tick 10, 3,692 at tick 30. The run was paused at tick 9 for about 11 h and resumed from its saved state; whether that played a part is not traced | **fail** (needs 90 exact, 0 wrong) |
| q-tesseract2-sum-ring-smoke-v2-1110 | 16-agent tesseract, ring, second attempt (host-checked) | 16/16 exact, 0 wrong, 0 no number; all held the total at tick 9 (the perfect-agent pace); 2 writes refused, both the wrong-base slip of the first attempt; every count answer 16 | smoke |
| **q-tesseract4-sum-ring-v2-1110** | tesseract side 4, ring, second attempt (host-checked) | **253/256 exact, 1 wrong, 2 no number** at eval_t40. First exact total at tick 17 (the scripted-agent pace), 83 agents held it that tick, 230 at tick 33, 253 at tick 40; never all 256. No agent ever held a wrong total; the one wrong answer (126,660) came from an agent that did not hold the total. 254 of 256 count answers were 256. 36 writes refused by the host, 13 missing replies in 10,240 turns; 814 s a tick; median prompt 1,982 tokens at tick 1, 3,628 at tick 10, 4,191 at tick 30 | **pass** (230 exact, at most 5 wrong, by tick 40), with the extra host help described below |

Baseline for the same model and recipe with flooding: 4x4 solved at tick 18 at 182 s a tick; 10x10 0/100 at tick 60 at
375 s a tick with median prompts of 4,080 tokens at tick 10 and 5,600 at tick 60.

Rows are filled from `.venv/bin/python experiments/tree_ring/measure.py runs/<run>`. The last two were filled 2026-11-10 15:40Z after the chain ended; Trace of the two-tree failure is below.

**Trace of the 27B two-tree run (`q-10x10-torus-sum-tree-verify-1110`), from ticks.jsonl.** Every parent added correctly
what was on its input at the time. The three losses are all a child changing its `sub` entry after its parent had already
used it and passed its own sum up; the parent never revised.
- Tree A, one short: agent 14 (number 121) read `sub 899 2` from its child at tick 12 and wrote `sub 1020 3`. The child's entry
  then changed (1099 at tick 13, 900 in the end). 900 + 121 = 1021.
- Tree B, 100 short: agent 69 (number 416) read `sub 1901 3` at tick 14 and wrote `sub 2317 4`; the child's entry was 2001
  from tick 15.
- Tree B, 1,726 and 4 agents short: agent 72 (number 631) read `sub 578 1` at tick 10 and wrote `sub 1209 2`; that child had
  reported before its own children came in and ended at `sub 2304 5`.
100 + 1,726 = 1,826 = 49,689 - 47,863, and 96 = 100 - 4, so these three account for both wrong totals. The host check added
in the first amendment did not refuse any write (0 rejected). All three happened at ticks 10 to 14, just after the run
resumed from its pause at tick 9; whether the resume caused the early or changed entries is not established. Not traced:
why the children's first entries were early or wrong.

## Ring arm: first attempt failed on both models; second amendment (2026-11-10 03:50Z)

- **Haiku, tesseract side 4 (`haiku-tesseract4-ring-1110`): fail.** 186/256 exact, 70 wrong, 0 no number at tick 40 (bar: 230
  exact, 5 wrong). 56 of the wrong answers are 129,797 = the true 129,219 + 578, one agent's number counted twice (46 of them
  with a count of 257), passed on faithfully round the rings. $14.68.
- **27B, 16-agent ring smoke (`q-tesseract2-sum-ring-smoke-1110`): fail.** 0/16 exact, 13 wrong, 3 no number; every agent held
  a total by tick 10 and none was right. The chain's gate then skipped the 256-agent run, so no card time went on it.
  What went wrong, from the record: after finishing ring d, agents kept passing on ring-d entries but added their *new*
  result (the base for ring d+1) instead of the base for ring d; and when a corrupted copy of their own entry came home
  later they overwrote the result they had saved.
- Both failures are the same kind as the tree's: one wrong addition is carried on faithfully and nothing downstream can
  tell. The tree arm has a second tree to catch it. The ring arm had nothing.

**Ring, second attempt (labelled v2 in run names).** Changes: the host records each agent's ring results itself (the first
entry of its own to come home on each ring, never changed) and shows them every tick, so the agent no longer keeps them in
notes; and the host refuses a ring write that breaks the stated rules (own entry not equal to the base; a passed-on
entry that is not on the input now, or not exactly input + base for that ring; a write before the base exists), saying
what it expected. The method text says so and spells out that the base for ring d does not change when ring d finishes.
With these checks a wrong total can only come from a host bug; what is left to test is whether the agents can drive it to
completion. This is more host help than the tree arm gets and has to be read that way. Scripted agents: 17 ticks on
side 4, 6/6 solved at 5% and at 20% missing replies, 0 rejected writes. The bar for v2 stays the frozen one
(230/256 exact, at most 5 wrong, by tick 40).

**Trace of the first Haiku tesseract run (by test, read-only).** The 578 double count: at tick 5 agent 82 added its base to the
entry it had itself put on q0 the tick before (shown to it in the decoded view of its own outputs) instead of to the entry
on p0; at tick 6 the entry's owner (agent 146) saw its id come home a second time and overwrote its correct ring-0 result
"since it can grow". That went into ring 1 and spread to a quarter of the network: 129,797. The other four wrong totals
are each one ring-2 or ring-3 write that does not equal input + base (base added twice, or the input added twice).
Replaying the v2 check over all 5,441 ring writes of that run refuses 14: those 5, one more wrong sum that never reached
an answer, and 7 entries passed on that were not on the input. 41 writes used `as="pairs"` on ring ports; no wrong total
traced to them and v2 does not refuse them (follow-up).

Haiku ring v2 was stopped (the fix goes to the 27B first): the 16-agent smoke
`haiku-tess2-ring-smoke-v2-1110` had all 16 holding the right total at tick 10 with 1 refused write when it was stopped; it
was never evaluated, so it is not a result.

## Third amendment: the once-only rule, enforced by the host (2026-11-10 ~18:00Z, bars frozen before any run)

The 27B two-tree run failed because agents broke two rules the method already states ("never write a sub before all
three have answered, and never change it afterwards"). The older literature's answer (Dijkstra and Scholten 1980; Chang
1982; Segall 1983) is that a node reports exactly once, after every child's report is final. With `tree_check = true` the
host now enforces that, as it enforces the ring rules in the second amendment:
- a sub goes only on the parent's output, only once the other three inputs all show an entry for that tree;
- a sub, once written, never changes (only that tree's total may replace it);
- the root writes its total only once all four inputs show an entry; any other agent's total must copy one on an input.
A refused write is not made and the agent is told why. The host does not check whether the sum is right: a sub that
adds the wrong entries is still accepted. One sentence is added to the method text saying the host checks. Perfect
scripted agents are never refused (10x10 and 20x20, one and two trees, 0%, 5% and 20% missed replies; same ticks as
before). Code: `tree_check_write` in `sim.py`; tests in `tests/test_tree_ring.py`.

| arm | world | pass bar |
|---|---|---|
| T-once smoke | 4x4 torus, two trees, roots 0 and 10, same seed | 16/16 exact, 0 wrong (gate: the 10x10 runs only if this passes) |
| T-once | 10x10 torus, two trees, roots 0 and 55, same seed | at least 90/100 exact by tick 60, 0 wrong (the T-verify bar) |

Qwen3.8-27B, medium thinking, the same recipe and prompt as `q-10x10-torus-sum-tree-verify-1110` apart from the sentence
above; served on seven Flare servers instead of four, which changes speed but not the prompts.
