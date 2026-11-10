"""Composite captions and labels over rendered frames, then encode the film.

usage: compose.py FRAMEDIR OUT.mp4 [--workers N] [--preview]
Shots are rendered on twos, so each output frame takes the nearest rendered frame at or before it.
"""
import argparse, glob, json, multiprocessing as mp, os, re, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = json.load(open(f"{ROOT}/timeline.json"))
FPS, W, H = 24, 1920, 1080
GREEN = (130, 194, 50)
LIGHT = (224, 224, 224)
F_CAP = ImageFont.truetype(f"{ROOT}/fonts/value-sans-regular.ttf", 44)
F_LAB = ImageFont.truetype(f"{ROOT}/fonts/value-sans-regular.ttf", 24)
F_END = ImageFont.truetype(f"{ROOT}/fonts/value-sans-regular.ttf", 34)
MAX = 58


def chunks(text):
    """Split a line into caption screens at sentence or clause breaks."""
    parts = re.split(r"(?<=[.?!:])\s+", text)
    out = []

    def split(p):
        if len(p) <= MAX:
            out.append(p)
            return
        # Cut near the middle, preferring a comma, so no screen is left with an orphaned word.
        mid = len(p) // 2
        cands = [i + 1 for i in range(len(p)) if p[i] == "," and abs(i - mid) < len(p) * .28]
        cut = min(cands, key=lambda i: abs(i - mid)) if cands else min((i for i in range(len(p)) if p[i] == " "), key=lambda i: abs(i - mid))
        split(p[:cut].strip())
        split(p[cut:].strip())

    for p in parts:
        split(p)
    # Merge little fragments forward.
    merged = []
    for p in out:
        if merged and len(merged[-1]) + len(p) + 1 <= MAX and (len(p) < 18 or len(merged[-1]) < 18):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


CAPS = []
for k, v in TL.items():
    if k == "_total" or not v.get("cap"):
        continue
    cs = chunks(v["cap"])
    total = sum(len(c) for c in cs)
    t = v["start"]
    for c in cs:
        d = v["dur"] * len(c) / total
        CAPS.append((t, t + d + .04, c))
        t += d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TREE_TICKS, held_at
LABEL_FROM = TL["b1"]["start"]
END_FROM = TL["h2"]["start"]
F_TICK = ImageFont.truetype(f"{ROOT}/fonts/value-sans-bold.ttf", 64)
F_TSUB = ImageFont.truetype(f"{ROOT}/fonts/value-sans-regular.ttf", 30)
# corner label per section: what is real on screen
SECTIONS = [("b1", "10×10 TORUS  /  QWEN3.8-27B  /  FLOOD PACKETS: ILLUSTRATIVE"),
            ("c1", "RUN q-10x10-torus-sum-tree-told  /  QWEN3.8-27B  /  TREE AND TICKS: REAL"),
            ("e1", "RUN haiku-10x10-tree-verify  /  CLAUDE HAIKU 5.5  /  BOTH TREES: REAL"),
            ("g1", "RUN haiku-4x28-arrays-tree  /  CLAUDE HAIKU 5.5  /  TREE AND TICKS: REAL"),
            ("f1", "TESSERACT SIDE 4  /  RESULT NUMBERS: REAL  /  WHICH MONKEYS: ILLUSTRATIVE"),
            ("h1", "EARLY RESEARCH  /  ONE SEED")]


def rounded(size, r, fill):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), r, fill=fill)
    return im


_cache = {}


def cap_layer(text):
    if text in _cache:
        return _cache[text]
    w = int(F_CAP.getlength(text))
    im = rounded((w + 76, 76), 6, (10, 12, 14, 188))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 9, 75), fill=(*GREEN, 255))
    d.text((44, 38), text, font=F_CAP, fill=(*LIGHT, 255), anchor="lm")
    _cache[text] = im
    return im


def label_layer(text):
    w = int(F_LAB.getlength(text))
    im = Image.new("RGBA", (w + 22, 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 3, 6, 27), fill=(*GREEN, 255))
    d.text((18, 15), text, font=F_LAB, fill=(*LIGHT, 225), anchor="lm")
    return im


def end_layer():
    text = "fourfold: tree and ring sums instead of flooding  /  one seed per run  /  early research"
    w = int(F_END.getlength(text))
    im = rounded((w + 60, 66), 6, (10, 12, 14, 200))
    ImageDraw.Draw(im).text((30, 33), text, font=F_END, fill=(*LIGHT, 255), anchor="lm")
    return im


LABELS = [(TL[i]["start"], label_layer(t)) for i, t in SECTIONS]
END = end_layer()
_tick = {}


def tick_layer(t):
    """Tick counter for the tree run, from the run's real ticks."""
    n, have = held_at(t)
    k = (n, have)
    if k not in _tick:
        im = rounded((470, 132), 8, (10, 12, 14, 196))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 9, 131), fill=(*GREEN, 255))
        d.text((34, 46), f"TICK {n}", font=F_TICK, fill=(*LIGHT, 255), anchor="lm")
        d.text((34, 104), f"EXACT TOTAL: {have} OF 100", font=F_TSUB, fill=(*(GREEN if have else LIGHT), 255), anchor="lm")
        _tick[k] = im
    return _tick[k]


def work(args):
    n, src, dst = args
    t = (n - 1) / FPS
    im = Image.open(src).convert("RGBA")
    if im.size != (W, H):
        im = im.resize((W, H), Image.BICUBIC)
    if LABEL_FROM <= t < END_FROM:  # below the caption band, clear of the signs
        lb = [l for st, l in LABELS if st <= t][-1]
        im.alpha_composite(lb, (W - lb.width - 24, H - lb.height - 8))
    if TREE_TICKS[1] <= t < TL["d3"]["start"]:
        im.alpha_composite(tick_layer(t), (40, 40))
    for a, b, c in CAPS:
        if a <= t < b:
            lay = cap_layer(c)
            im.alpha_composite(lay, ((W - lay.width) // 2, H - lay.height - 44))
            break
    if t >= END_FROM:
        im.alpha_composite(END, ((W - END.width) // 2, 24))
    im.convert("RGB").save(dst, compress_level=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tmp", default=f"{ROOT}/out/comp")
    ap.add_argument("--crf", default="19")
    ap.add_argument("--audio", default=f"{ROOT}/out/mix.wav")
    a = ap.parse_args()
    have = sorted(int(os.path.basename(f)[1:7]) for f in glob.glob(f"{a.frames}/f*.png"))
    os.makedirs(a.tmp, exist_ok=True)
    total = round(TL["_total"] * FPS)
    jobs = []
    idx = 0
    for n in range(1, total + 1):
        while idx + 1 < len(have) and have[idx + 1] <= n:
            idx += 1
        dst = f"{a.tmp}/c{n:06d}.png"
        jobs.append((n, f"{a.frames}/f{have[idx]:06d}.png", dst))
    with mp.Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, jobs, 8)):
            if i % 500 == 0:
                print("comp", i, flush=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", f"{a.tmp}/c%06d.png", "-i", a.audio,
           "-c:v", "libx264", "-preset", "slow", "-crf", a.crf, "-pix_fmt", "yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print("COMPOSE_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
