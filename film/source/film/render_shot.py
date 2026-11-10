"""Render one shot.

env: SHOT=s05  OUTDIR=...  W H SPP  STEP=2 (on-twos)  PART=i/n (split frames across workers)
     ONLY=frame,frame (specific global frames)  BUILD_ONLY=1 (build only, print range)
"""
import bpy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(bpy.data.filepath)))
import shots

name = os.environ["SHOT"]
a, b = shots.SHOTS[name]()
sc = bpy.context.scene
print(f"SHOT {name} frames {a}..{b}", flush=True)
if os.environ.get("BUILD_ONLY"):
    sys.exit(0)
out = os.environ["OUTDIR"]
os.makedirs(out, exist_ok=True)
step = int(os.environ.get("STEP", 2))
frames = list(range(a, b, step))
if os.environ.get("ONLY"):
    frames = [int(x) for x in os.environ["ONLY"].split(",")]
i, n = (int(x) for x in os.environ.get("PART", "0/1").split("/"))
frames = frames[i::n]
for f in frames:
    path = f"{out}/f{f:06d}.png"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    if os.environ.get("CLAIM"):
        try:
            os.mkdir(f"{out}/.claim_{f:06d}")
        except FileExistsError:
            continue
    t = time.time()
    sc.frame_set(f)
    sc.render.filepath = path + ".tmp.png"
    bpy.ops.render.render(write_still=True)
    os.replace(path + ".tmp.png", path)
    print(f"DONE {name} {f} {time.time() - t:.1f}s", flush=True)
print(f"SHOT_COMPLETE {name} part {i}/{n} {len(frames)} frames", flush=True)
