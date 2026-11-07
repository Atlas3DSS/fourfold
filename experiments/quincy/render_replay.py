"""replay.gif in the style of erz0r's examples/sol-50/replay.gif, for any grid size or a torus.

It uses view.py's own App.draw and render.py's ScreenRenderer, so every tile is drawn exactly as the
original viewer draws it. Like the original GIF, the frame is the square tile grid with the status bar
cropped off, each tick shows for ~0.34 s (the original: 172 frames of 100 ms for 50 ticks), and it
covers the first 50 ticks.

  python experiments/quincy/render_replay.py runs/<run> out.gif [--ticks 1-50] [--tile 180] [--step N]

--tile is the pixel size per tile (the original: 540 px for 3 tiles = 180 px, gaps included).
--step keeps every Nth tick, for long runs or very large grids.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

import view  # noqa: E402
from render import ScreenRenderer  # noqa: E402
from runlog import Run  # noqa: E402

TICK_MS = 340


def open_run(path: Path) -> tuple[Run, Path | None]:
    """runlog.Run reads a plain ticks.jsonl; a .gz copy is unpacked to a temp dir first."""
    f = path / "ticks.jsonl" if path.is_dir() else path
    if f.exists():
        return Run(f), None
    gz = path / "ticks.jsonl.gz"
    tmp = Path(tempfile.mkdtemp(prefix="ff-replay-"))
    with gzip.open(gz, "rb") as src, open(tmp / "ticks.jsonl", "wb") as dst:
        shutil.copyfileobj(src, dst)
    return Run(tmp / "ticks.jsonl"), tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--ticks", default="1-50", help="first-last tick (default 1-50, like the original)")
    ap.add_argument("--tile", type=int, default=180, help="pixels per tile (original: 180)")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth tick")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="draw at this multiple of the output size, then scale down (the original reads as 2x)")
    args = ap.parse_args()

    pygame.init()
    run, tmp = open_run(Path(args.run))
    run.poll()
    rows, cols = run.meta["rows"], run.meta["cols"]
    W = round(args.tile * cols * args.scale)
    H = round(args.tile * rows * args.scale)
    out_size = (args.tile * cols, args.tile * rows)
    app = view.App(run, ScreenRenderer(), live=False)
    sfont = app.renderer.font(max(10, min(16, max(W, H) // 70)))
    sb = sfont.get_linesize() + 6            # view.draw reserves this for its status bar; we crop it off
    lo, hi = (int(x) for x in args.ticks.split("-"))
    idxs = [i for i, r in enumerate(run.ticks) if lo <= r["t"] <= hi][::args.step]
    if not idxs:
        sys.exit(f"no ticks in {args.ticks} (run has {run.ticks[0]['t']}..{run.ticks[-1]['t']})")

    frames = []
    # like the original: draw on a WxH window (the status bar takes its share of the height), crop the bar
    # off, and scale the rest to the output size
    surf = pygame.Surface((W, H))
    for i in idxs:
        app.idx = i
        app.cache.clear()
        app.draw(surf)
        raw = pygame.image.tostring(surf, "RGB")
        img = Image.frombytes("RGB", (W, H), raw).crop((0, 0, W, H - sb))
        if img.size != out_size:
            img = img.resize(out_size, Image.LANCZOS)
        frames.append(img.quantize(colors=64, dither=Image.Dither.NONE))
        print(f"\rtick {run.ticks[i]['t']}", end="", flush=True)
    print()
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=TICK_MS * args.step,
                   loop=0, optimize=True, disposal=1)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out}: {out_size[0]}x{out_size[1]}, {len(frames)} frames, {Path(args.out).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
