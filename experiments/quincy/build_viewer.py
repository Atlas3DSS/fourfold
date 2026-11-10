"""build the torus viewer page from one or more runs.

  python experiments/quincy/build_viewer.py [--theme theme.css] OUT.html "label=RUN_DIR" ["label=RUN_DIR@source"] ...
  python experiments/quincy/build_viewer.py [--theme theme.css] --split SITE_DIR "label=RUN_DIR" ...

RUN_DIR@K traces influence from agent K (0-based) as well.
--notes FILE: hand-written {"<run>": {"title", "note", "status"}} for the "about this run" panel (run_summary.py).

default: one self-contained page with every run embedded.
--split: SITE_DIR/index.html carries only the run list; each run is SITE_DIR/data/<run>.json (+ .json.gz), fetched
when the viewer picks it. a run whose data file is newer than its ticks.jsonl and eval files is not rebuilt.
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_data import build  # noqa: E402

args = sys.argv[1:]
theme = ""
if "--theme" in args:                     # a css file whose :root block overrides the viewer's tokens
    i = args.index("--theme")
    theme = Path(args[i + 1]).read_text(encoding="utf-8")
    del args[i:i + 2]
notes, notes_mtime = {}, 0.0
if "--notes" in args:
    i = args.index("--notes")
    nf = Path(args[i + 1])
    if nf.exists():
        notes, notes_mtime = json.loads(nf.read_text(encoding="utf-8")), nf.stat().st_mtime
    del args[i:i + 2]
split = None
if "--split" in args:
    i = args.index("--split")
    split = Path(args[i + 1])
    del args[i:i + 2]
    specs = args
else:
    out, specs = Path(args[0]), args[1:]


def parse(spec):
    label, _, path = spec.partition("=")
    path, _, src = path.partition("@")
    return label, Path(path), (int(src) if src else None)


HERE = Path(__file__).resolve().parent
CODE_MTIME = max((HERE / n).stat().st_mtime for n in ("viz_data.py", "run_summary.py"))   # new code = rebuild the data too


def newest_input(run: Path) -> float:
    files = [f for f in [run / "ticks.jsonl", run / "ticks.jsonl.gz", run / "ratchet.jsonl", *run.glob("eval-t*.json")] if f.exists()]
    return max(CODE_MTIME, *(f.stat().st_mtime for f in files))


sets = []
if split:
    data_dir = split / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    keep = set()
    for spec in specs:
        label, run, src = parse(spec)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", run.name + (f"@{src}" if src is not None else "")) + ".json"
        f = data_dir / name
        keep.update({name, name + ".gz"})
        if not f.exists() or f.stat().st_mtime < max(newest_input(run), notes_mtime):
            blob = json.dumps(build(run, src, notes.get(run.name)), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            tmp = f.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(f)
            gz = data_dir / (name + ".gz")
            gz.with_suffix(".gztmp").write_bytes(gzip.compress(blob, 6))
            gz.with_suffix(".gztmp").replace(gz)
        sets.append({"label": label, "url": f"data/{name}", "kb": f.stat().st_size // 1024})
    for old in data_dir.iterdir():           # runs that left the list
        if old.name not in keep and not old.name.startswith("film-"):   # film-*.json: a film for the banner, put there by hand
            old.unlink()
    out = split / "index.html"
else:
    for spec in specs:
        label, run, src = parse(spec)
        sets.append({"label": label, "data": build(run, src, notes.get(run.name))})

tpl = (Path(__file__).resolve().parent / "viewer" / "torus.html").read_text(encoding="utf-8")
blob = json.dumps(sets, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
banner = notes.get("_banner") if isinstance(notes, dict) else None   # optional site-wide highlight: {"html": "..."} (trusted, hand-written)
banner_html = ""
if isinstance(banner, dict) and banner.get("html"):   # with "summary": one line that folds out to the full text
    banner_html = (f'<details class="banner"><summary>{banner["summary"]}</summary><div class="banner-body">{banner["html"]}</div></details>'
                   if banner.get("summary") else f'<div class="banner">{banner["html"]}</div>')
page = (tpl.replace("/*THEME*/", theme).replace("/*BUILT*/0", str(int(time.time())))
        .replace("/*DATASETS*/", blob).replace("<!--BANNER-->", banner_html))
tmp = out.with_suffix(".tmp")
tmp.write_text(page, encoding="utf-8")
tmp.replace(out)
print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(sets)} runs{', split' if split else ''})")
