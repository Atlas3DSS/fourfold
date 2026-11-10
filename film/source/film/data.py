"""Run data and the narration clock, shared by shots.py and compose.py.

tree10.json          q-10x10-torus-sum-tree-told-1110 (Qwen3.8-27B, one tree, root agent 0): 100/100 exact at tick 33.
tree10_two_broken    haiku-10x10-tree-verify-1110 (Claude Haiku 5.5, two trees): tree B lost agent 63's branch of 11.
Made by film/treedata.py from fourfold/runs.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = json.load(open(f"{ROOT}/timeline.json"))
sys.path.insert(0, ROOT)
from narration import LINES
SPOKEN = {i: s for i, s, c, g in LINES if s}


def _load(name):
    d = json.load(open(f"{ROOT}/data/{name}.json"))
    for t in d["trees"].values():
        for k in ("join", "sub", "total", "parent"):
            t[k] = {int(a): b for a, b in t[k].items()}
    return d


T10 = _load("tree10")
BROKEN = _load("tree10_two_broken")
TOTAL10 = T10["answer"]          # 49,689


def word_t(i, word, off=0.):
    """Seconds at which `word` starts in line i, estimated by character position."""
    s = SPOKEN[i]
    k = s.index(word)
    return TL[i]["start"] + TL[i]["dur"] * k / len(s) + off


def _piecewise(anchors):
    out = {}
    for (t0, s0), (t1, s1) in zip(anchors, anchors[1:]):
        for t in range(t0, t1 + 1):
            out[t] = s0 + (s1 - s0) * (t - t0) / (t1 - t0)
    return out


# The 27B tree run, ticks 1..33 (+1 so a bead leaving at the last tick has somewhere to land): the wave goes out under c2,
# the sums climb under c3 and c4, the first exact total is at tick 23 and the last at tick 33, as the narration says.
TREE_TICKS = _piecewise([(1, TL["c2"]["start"] + .5), (21, TL["c4"]["start"]), (23, word_t("c4", "Then the answer")),
                         (33, word_t("d1", "Tick thirty-three")), (34, word_t("d1", "Tick thirty-three") + .7)])

def held_at(t):
    """How many of the 100 hold the exact total at time t (s), and the tick shown."""
    n = max([k for k, v in TREE_TICKS.items() if v <= t] or [0])
    n = min(n, 33)
    return n, sum(1 for v in T10["trees"]["A"]["total"].values() if v[0] <= n)
