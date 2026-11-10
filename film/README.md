# Pass It Up

A two-and-a-half minute film of section 11 of [`EXPERIMENTS-QUINCY.md`](../EXPERIMENTS-QUINCY.md): why flooding stopped
working at 100 agents, the tree sum that replaced it, the branch a tree can silently lose, and the ring sum that broke on
the 256-agent tesseract. It is a communications piece, not a record: the record is [`TREE-RING-1109.md`](../TREE-RING-1109.md).

- `Pass_It_Up_web.mp4`: 720p, with burned-in captions. `Pass_It_Up.srt`: the captions. `poster.jpg`: a still.
- `source/`: the script (`narration.py`), the Blender scene code (`film/shots.py`, `film/lib.py`), the score and sound
  (`film/audio.py`, all synthesised in that file), the compositor, and the run data the animation is driven by
  (`data/*.json`, made by `film/treedata.py` from the run records).

What on screen is real and what is not:
- The tree in the 100-agent scenes is the real one from `q-10x10-torus-sum-tree-told-1110`: who took whom as parent, the
  tick each agent joined, the tick it sent its sum up, and the tick it held the total. The tick counter is the run's.
- The broken branch is the real one from `haiku-10x10-tree-verify-1110` (agent 63's branch of 11 in the second tree).
- The swarm scene plays the real tree of `haiku-4x28-arrays-tree-1110`.
- Flood packets are illustrative. In the tesseract scene the totals (186 of 256 exact, 70 wrong) are real; which monkeys
  are shown red is illustrative.

The source does not build on its own: the stage, the monkey rig's materials, the fonts and the voice model are house
assets that are not in this repository. The voice is synthesised (Kokoro). Nobody has listened to the audio mix or
watched the film in motion before publication: it was checked as rendered stills and by measurement only.
