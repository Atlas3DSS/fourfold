"""embed one or more runs into the torus viewer as a single self-contained page.

  python experiments/quincy/build_viewer.py [--theme theme.css] OUT.html "label=RUN_DIR" ["label=RUN_DIR@source"] ...

RUN_DIR@K traces influence from agent K (0-based) as well.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_data import build  # noqa: E402

args = sys.argv[1:]
theme = ""
if "--theme" in args:                     # a css file whose :root block overrides the viewer's tokens
    i = args.index("--theme"); theme = Path(args[i + 1]).read_text(encoding="utf-8"); del args[i:i + 2]
out, specs = Path(args[0]), args[1:]
sets = []
for spec in specs:
    label, _, path = spec.partition("=")
    path, _, src = path.partition("@")
    sets.append({"label": label, "data": build(Path(path), int(src) if src else None)})
tpl = (Path(__file__).resolve().parent / "viewer" / "torus.html").read_text(encoding="utf-8")
blob = json.dumps(sets, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
out.write_text(tpl.replace("/*THEME*/", theme).replace("/*DATASETS*/", blob), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(sets)} runs)")
