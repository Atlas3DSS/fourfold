"""map.png in the style of erz0r's examples/sol-50/map.png, for any grid size or a torus.

Left: the best map an agent drew in its final answer (highest map score in the run's eval). Right: the true
wiring, turned and shifted the same way eval.py matched it, with a tick on every agent placed right and a cross
on every agent placed wrong. Middle: placed right / total. The titles say what the agents were told, so a
primed run is never captioned as unprompted.

  python experiments/quincy/render_map.py runs/<run> out.png [--agent K]     (K 1-based; default: best map)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

BG, TILE, EDGE = (0, 0, 0), (14, 14, 14), (52, 52, 52)
FG, MUTED, AMBER, WRONG = (236, 236, 236), (130, 130, 130), (255, 176, 48), (150, 70, 60)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(px, bold=False):
    return ImageFont.truetype(MONO_B if bold else MONO, px)


def load(run: Path):
    f = run / "ticks.jsonl"
    fh = open(f, encoding="utf-8") if f.exists() else gzip.open(run / "ticks.jsonl.gz", "rt", encoding="utf-8")
    with fh:
        meta = json.loads(fh.readline())
    evals = sorted(run.glob("eval-t*.json"), key=lambda p: int(p.stem.split("-t")[1]))
    if not evals:
        sys.exit(f"{run}: not scored yet (no eval-t*.json)")
    return meta, json.loads(evals[-1].read_text())


def best_transform(cells: dict, meta: dict):
    """the same search as eval.score_map, returning the winning map from drawn cell -> true position."""
    R, C = meta["rows"], meta["cols"]
    truth = {tuple(a["pos"]): a["seed"] for a in meta["agents"]}
    torus = meta.get("topology") == "torus"
    best = None
    for swap in (False, True):
        for fr in (1, -1):
            for fc in (1, -1):
                pts = {}
                for (r, c), v in cells.items():
                    rr, cc = (c, r) if swap else (r, c)
                    pts[(r, c)] = (fr * rr, fc * cc)
                r0 = min(p[0] for p in pts.values()); c0 = min(p[1] for p in pts.values())
                h = max(p[0] for p in pts.values()) - r0 + 1; w = max(p[1] for p in pts.values()) - c0 + 1
                if torus and (h > R or w > C):
                    continue
                for dr in range(R if torus else R - h + 1):
                    for dc in range(C if torus else C - w + 1):
                        def at(cell, swap=swap, fr=fr, fc=fc, r0=r0, c0=c0, dr=dr, dc=dc):
                            r, c = cell
                            rr, cc = (c, r) if swap else (r, c)
                            rr, cc = fr * rr - r0 + dr, fc * cc - c0 + dc
                            return (rr % R, cc % C) if torus else (rr, cc)
                        ok = sum(1 for k, v in cells.items() if v and truth.get(at(k)) == v)
                        bad = sum(1 for k, v in cells.items() if v and truth.get(at(k)) != v)
                        key = (max(0.0, (ok - bad) / len(truth)), ok)
                        if best is None or key > best[0]:
                            best = (key, at, ok, bad, swap)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--agent", type=int, help="1-based agent whose map to show (default: the best map)")
    args = ap.parse_args()
    meta, ev = load(Path(args.run))
    R, C, N = meta["rows"], meta["cols"], len(meta["agents"])
    torus = meta.get("topology") == "torus"
    truth = {tuple(a["pos"]): a["seed"] for a in meta["agents"]}
    pos_of = {a["seed"]: tuple(a["pos"]) for a in meta["agents"]}
    res = ev["results"]
    drawn = [r for r in res if r["map_cells"]]
    pick = (next(r for r in res if r["agent"] == args.agent) if args.agent else
            max(drawn, key=lambda r: (r["map_score"], r["map_correct"])) if drawn else None)
    cells = {tuple(int(x) for x in k.split(",")): v for k, v in (pick["map_cells"] if pick else {}).items()}
    best = best_transform(cells, meta) if cells else None

    # one frame for both panels: the drawn map's orientation, extended to the whole true grid
    if best:
        _, at, ok, bad, swap = best
        frame = {}                       # drawn-frame cell -> true position
        rmin = min(r for r, _ in cells); cmin = min(c for _, c in cells)
        if torus:                        # one full period starting at the drawing's corner: every agent once
            fr_n, fc_n = (C, R) if swap else (R, C)
            rows_, cols_ = range(rmin, rmin + fr_n), range(cmin, cmin + fc_n)
        else:                            # a window big enough to hold the whole grid around the drawing
            rows_ = range(rmin - max(R, C), rmin + 2 * max(R, C)); cols_ = range(cmin - max(R, C), cmin + 2 * max(R, C))
        for r in rows_:
            for c in cols_:
                p = at((r, c))
                if p in truth and p not in frame.values():
                    frame[(r, c)] = p
        r0 = min(r for r, _ in frame); c0 = min(c for _, c in frame)
        frame = {(r - r0, c - c0): p for (r, c), p in frame.items()}
        cells = {(r - r0, c - c0): v for (r, c), v in cells.items()}
    else:
        ok = bad = 0
        frame = {(r, c): (r, c) for r in range(R) for c in range(C)}
    FR = max(r for r, _ in frame) + 1; FC = max(c for _, c in frame) + 1

    big = max(FR, FC) <= 4
    tile = 128 if big else max(44, min(84, 520 // max(FR, FC)))
    gap = 16 if big else max(5, tile // 9)
    panel_w = FC * tile + (FC - 1) * gap; panel_h = FR * tile + (FR - 1) * gap
    W = max(1600, 2 * panel_w + 300 + 2 * 150); H = max(900, 270 + panel_h + 215)
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    s = W / 1600

    # titles: say exactly what the agents were told
    prime = meta["config"].get("prime") or ""
    shape = f"{R}×{C} {'torus' if torus else 'grid'}"
    title = (f"{N} llms on a {shape}. none were told the others exist." if not prime
             else f"{N} llms on a {shape}. each was told only what's below.")
    t_end = ev["tick"]
    if pick:
        sub = (f"after {t_end} ticks, agent {pick['agent']} drew the best map: {ok} of {N} placed right."
               if ok < N else f"after {t_end} ticks, agent {pick['agent']} drew the whole map.")
    else:
        alone = sum(r["count"] == 1 for r in res)
        sub = f"after {t_end} ticks, no agent drew a map. {alone} of {N} said they were alone."
    d.text((80 * s, 62), title, font=font(round(32 * s), True), fill=FG)
    d.text((80 * s, 112), sub, font=font(round(26 * s)), fill=MUTED)

    def draw_panel(x0, label, ids, marks):
        lf = font(round(20 * s))
        lw = d.textlength(label, font=lf)
        d.text((x0 + panel_w / 2 - lw / 2, 222), label, font=lf, fill=MUTED)
        y0 = 270
        idf = font(max(9, min(20, round(tile * 0.155))))
        for (r, c), v in ids.items():
            x, y = x0 + c * (tile + gap), y0 + r * (tile + gap)
            for (r2, c2) in ((r, c + 1), (r + 1, c)):          # connectors to drawn/true neighbours
                if (r2, c2) in ids:
                    if c2 > c:
                        d.line((x + tile, y + tile / 2, x + tile + gap, y + tile / 2), fill=EDGE, width=2)
                    else:
                        d.line((x + tile / 2, y + tile, x + tile / 2, y + tile + gap), fill=EDGE, width=2)
            d.rounded_rectangle((x, y, x + tile, y + tile), radius=max(3, tile // 32), fill=TILE, outline=EDGE)
            if v:
                col = AMBER if pick and v == pick["seed"] else FG
                tw = d.textlength(v, font=idf)
                d.text((x + tile / 2 - tw / 2, y + tile / 2 - idf.size * 0.6), v, font=idf, fill=col)
            m = marks.get((r, c))
            if m:
                mf = font(max(9, round(tile * 0.17)), True)
                d.text((x + tile - tile * 0.2, y + tile * 0.04), m[0], font=mf, fill=m[1])

    lx = 150 * s; rx = W - 150 * s - panel_w
    left_label = f"agent {pick['agent']}'s final answer" if pick else "no map drawn"
    draw_panel(lx, left_label, cells if pick else {}, {})
    right = {k: truth[p] for k, p in frame.items()}
    marks = {}
    for k, v in cells.items():
        if v:
            marks[k] = ("✓", AMBER) if right.get(k) == v else ("×", WRONG)
    draw_panel(rx, "the true wiring", right, marks)

    sf = font(round(56 * s), True)
    score = f"{ok} / {N}"
    sw = d.textlength(score, font=sf)
    d.text((W / 2 - sw / 2, 270 + panel_h / 2 - 60), score, font=sf, fill=AMBER)
    pf = font(round(22 * s)); pw = d.textlength("placed right", font=pf)
    d.text((W / 2 - pw / 2, 270 + panel_h / 2 + 20), "placed right", font=pf, fill=MUTED)

    model = meta["config"].get("llm", {}).get("model") or meta["config"].get("agent")
    think = meta["config"].get("llm", {}).get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking")
    foot1 = ((f"true {'torus' if torus else 'grid'} shown {'turned and shifted' if torus else 'reflected'} to line up: "
              "no agent can know which way is north." + (" on a torus no cell is special." if torus else ""))
             if pick else f"true {'torus' if torus else 'grid'} as wired" + (": every edge wraps round." if torus else "."))
    foot2 = (f"{model} × {N}  ·  four {meta['config'].get('port_bits', 128)}-bit in/out ports each, shuffled"
             f"  ·  thinking {'off' if think is False else 'on'}  ·  {t_end} ticks")
    ff = font(round(18 * s))
    d.text((80 * s, H - 110), foot1, font=ff, fill=MUTED)
    d.text((80 * s, H - 76), foot2, font=ff, fill=MUTED)
    if prime:
        d.text((80 * s, H - 42), f"told: “{prime}”", font=ff, fill=AMBER)
    img.save(args.out)
    print(f"wrote {args.out} ({W}x{H}); map from agent {pick['agent'] if pick else '-'}: {ok} right, {bad} wrong")


if __name__ == "__main__":
    main()
