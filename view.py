"""pygame viewer for a run's ticks.jsonl. replays a finished run or tails a live one.

  python view.py                      # newest run, replay from tick 1
  python view.py <run> --live         # jump to the newest tick and keep following
  python view.py <run> --snapshot out.png --tick 40 --debug --protocol

keys: space play/pause   left/right step   shift+left/right jump 10
      home/end first/last   +/- speed   1-9 zoom into tile   0 grid
      p protocol   d debug overlay   esc/q quit
"""
from __future__ import annotations

import argparse
import os
import sys

import pygame

from render import SIDES, ScreenRenderer, TileView
from runlog import Run, resolve

BG = (0, 0, 0)
STATUS = (110, 110, 110)
LOG_LINES_KEPT = 400


class App:
    def __init__(self, run: Run, renderer, live: bool):
        self.run = run
        self.renderer = renderer
        self.idx = max(0, len(run.ticks) - 1) if live else 0
        self.playing = live
        self.speed = 4.0
        self.acc = 0.0
        self.zoom: int | None = None
        self.show_protocol = False
        self.debug = False
        self.cache: dict = {}

    # -------------------------------------------------------------- model

    def tile_view(self, k: int, idx: int) -> TileView:
        run = self.run
        a = run.meta["agents"][k]
        rec = run.ticks[idx]
        ag = rec["agents"][k]
        prev_ag = run.ticks[idx - 1]["agents"][k] if idx > 0 else None
        prev = prev_ag["outputs"] if prev_ag else ["0" * len(v) for v in ag["outputs"]]
        prev_in = prev_ag["inputs"] if prev_ag else ["0" * len(v) for v in ag["inputs"]]
        sides, prev_sides, inputs, prev_inputs, side_ports, echo = {}, {}, {}, {}, {}, {}
        for i, side in enumerate(a["port_sides"]):
            side_ports[side] = i
            live = a["links"][i] is not None
            sides[side] = ag["outputs"][i] if live else None
            prev_sides[side] = prev[i] if live else None
            inputs[side] = ag["inputs"][i] if live else None
            prev_inputs[side] = prev_in[i] if live else None
            if self.debug and live:
                echo[side] = (run.echo(k, i, idx), run.echo_avg(k, i, idx))
        log = []
        j = idx
        while j >= 0 and len(log) < LOG_LINES_KEPT:
            r = run.ticks[j]
            x = r["agents"][k]
            chunk = [(r["t"], "log", l) for l in x["log"]] + [(r["t"], "error", e) for e in x["errors"]]
            log[:0] = chunk
            j -= 1
        return TileView(
            label=f"agent {k + 1}", seed=a["seed"], tick=rec["t"],
            sides=sides, side_ports=side_ports, prev_sides=prev_sides, echo=echo,
            inputs=inputs, prev_inputs=prev_inputs,
            log=log, protocol=ag["protocol"],
            show_protocol=self.show_protocol, debug=self.debug,
            pixel=int(run.meta["config"].get("pixel_size", 9)) if run.meta["config"].get("port_mode") == "pixels" else 0,
            score=ag.get("score"), score_max=sum(l is not None for l in a["links"]),
        )

    def tile_surface(self, k: int, size: tuple[int, int]) -> pygame.Surface:
        key = (k, self.idx, size, self.show_protocol, self.debug)
        surf = self.cache.get(key)
        if surf is None:
            if len(self.cache) > 300:
                self.cache.clear()
            surf = self.cache[key] = self.renderer.render(self.tile_view(k, self.idx), size)
        return surf

    # -------------------------------------------------------------- input

    def step(self, n: int):
        last = len(self.run.ticks) - 1
        self.idx = max(0, min(last, self.idx + n))

    def key(self, ev) -> bool:
        k, shift = ev.key, bool(ev.mod & pygame.KMOD_SHIFT)
        if k in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if k == pygame.K_SPACE:
            self.playing = not self.playing
            self.acc = 0.0
        elif k == pygame.K_RIGHT:
            self.step(10 if shift else 1)
        elif k == pygame.K_LEFT:
            self.step(-10 if shift else -1)
        elif k == pygame.K_HOME:
            self.idx = 0
        elif k == pygame.K_END:
            self.step(10 ** 9)
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(60.0, self.speed * 1.5)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(0.25, self.speed / 1.5)
        elif k == pygame.K_p:
            self.show_protocol = not self.show_protocol
        elif k == pygame.K_d:
            self.debug = not self.debug
        elif k in (pygame.K_0, pygame.K_KP0):
            self.zoom = None
        elif pygame.K_1 <= k <= pygame.K_9 or pygame.K_KP1 <= k <= pygame.K_KP9:
            n = k - (pygame.K_1 if k <= pygame.K_9 else pygame.K_KP1)
            if n < self.run.n_agents:
                self.zoom = n
        return True

    def update(self, dt: float):
        if not self.playing or not self.run.ticks:
            return
        self.acc += dt * self.speed
        while self.acc >= 1.0:
            self.acc -= 1.0
            self.step(1)   # at the end this just waits for new ticks

    # -------------------------------------------------------------- drawing

    def draw(self, screen: pygame.Surface):
        screen.fill(BG)
        W, H = screen.get_size()
        sfont = self.renderer.font(max(10, min(16, H // 70)))
        sb = sfont.get_linesize() + 6
        if not self.run.meta or not self.run.ticks:
            msg = sfont.render("waiting for ticks ...", True, STATUS)
            screen.blit(msg, (16, 16))
            return
        rows, cols = self.run.meta["rows"], self.run.meta["cols"]
        g = max(6, int(min(W, H) * 0.018))
        aw, ah = W, H - sb   # area above the status bar; everything is square and centred in it
        if self.zoom is not None:
            side = min(aw, ah) - 2 * g
            if side > 0:
                screen.blit(self.tile_surface(self.zoom, (side, side)),
                            ((aw - side) // 2, (ah - side) // 2))
        else:
            t = min((aw - (cols + 1) * g) // cols, (ah - (rows + 1) * g) // rows)
            if t > 0:
                x0 = (aw - (cols * t + (cols - 1) * g)) // 2
                y0 = (ah - (rows * t + (rows - 1) * g)) // 2
                for k in range(self.run.n_agents):
                    r, c = self.run.meta["agents"][k]["pos"]
                    screen.blit(self.tile_surface(k, (t, t)),
                                (x0 + c * (t + g), y0 + r * (t + g)))
        t = self.run.ticks[self.idx]["t"]
        last = self.run.ticks[-1]["t"]
        state = "playing" if self.playing else "paused"
        bits = [self.run.meta["run_id"], f"tick {t}/{last}", f"{state} {self.speed:g}/s",
                "zoom " + str(self.zoom + 1) if self.zoom is not None else "grid"]
        sc = [a["score"] for a in self.run.ticks[self.idx]["agents"] if "score" in a]
        if sc:
            best = sum(l is not None for a in self.run.meta["agents"] for l in a["links"])
            bits.append(f"score {sum(sc)}/{best}")
        if self.debug:
            bits.append("debug")
        text = "   ".join(bits) + "      space  <- ->  1-9 0  p  d  +/-"
        screen.blit(sfont.render(text, True, STATUS), (g, H - sb + 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run id, run dir or ticks.jsonl (default: newest run)")
    ap.add_argument("--live", action="store_true", help="start at the newest tick and play along")
    ap.add_argument("--size", default=None, help="window size WxH")
    ap.add_argument("--snapshot", metavar="PNG", help="render one frame to a file and exit")
    ap.add_argument("--tick", type=int, help="tick to show (snapshot or start)")
    ap.add_argument("--zoom", type=int, help="start zoomed into tile 1-9")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--protocol", action="store_true")
    args = ap.parse_args()

    if args.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    run = Run(resolve(args.run))
    run.poll()

    app = App(run, ScreenRenderer(), live=args.live)
    app.debug, app.show_protocol = args.debug, args.protocol
    if args.zoom:
        app.zoom = args.zoom - 1
    if args.tick is not None and run.ticks:
        app.idx = next((i for i, r in enumerate(run.ticks) if r["t"] >= args.tick), len(run.ticks) - 1)

    if args.size:
        size = tuple(int(x) for x in args.size.lower().split("x"))
    elif args.snapshot:
        size = (1920, 1080)
    else:
        dw, dh = pygame.display.get_desktop_sizes()[0]
        size = (int(dw * 0.85), int(dh * 0.85))

    if args.snapshot:
        surf = pygame.Surface(size)
        app.draw(surf)
        pygame.image.save(surf, args.snapshot)
        print(f"wrote {args.snapshot}")
        return

    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption(f"fourfold - {run.meta['run_id'] if run.meta else run.path}")
    pygame.key.set_repeat(300, 40)
    clock = pygame.time.Clock()
    poll_t = 0.0
    while True:
        dt = clock.tick(30) / 1000
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and not app.key(ev)):
                pygame.quit()
                return
            if ev.type == pygame.VIDEORESIZE:
                app.cache.clear()
        poll_t += dt
        if poll_t > 0.25:
            poll_t = 0.0
            run.poll()
        app.update(dt)
        app.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    sys.exit(main())
