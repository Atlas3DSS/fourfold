"""Pass It Up: all shots. Each builds one scene on the Atlas stage and keys it on global frames.

The trees, the ticks at which each monkey joined, reported and held the total, and every number on screen come from the
run records (data.py). Flood packets in the two flooding scenes are illustrative.
"""
import bpy, bmesh, math, os, random, sys
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from data import *
from data import _load


# ---------------------------------------------------------------- house props (from The Right Tool)
def sign(lines, loc, pal, size=2.4, fg="green", bg="dark", pad=1.2, w=None, h=None, parent=None, tilt=0., rx=0.):
    n = lines.split("\n")
    root = empty(f"sign_{n[0][:8]}", loc, parent)
    tw = w or (max(len(x) for x in n) * size * .66 + pad * 2)
    th = h or (len(n) * size * 1.25 + pad * 1.4)
    rbox("plate", (tw, .7, th), (0, 0, 0), pal[bg], root, bevel=.16)
    text(lines, (0, -.46, 0), size, pal[fg], root, extrude=.05)
    root.rotation_euler = (math.radians(rx), math.radians(tilt), 0)
    return root


def ring(loc, r, m, parent=None, rot=(math.pi / 2, 0, 0), minor=.085):
    bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=r * minor, major_segments=48, minor_segments=10,
                                     location=loc, rotation=rot)
    o = bpy.context.object
    bpy.ops.object.shade_smooth()
    o.data.materials.append(m)
    if parent:
        o.parent = parent
    return o


def card(label, pal, w=5.2, h=3.4, loc=(0, 0, 0), parent=None, size=.62, bg="light", fg="ink"):
    root = empty(f"card_{label[:6]}", loc, parent)
    rbox("c", (w, .14, h), (0, 0, 0), pal[bg], root, bevel=.05)
    text(label, (0, -.09, 0), size, pal[fg], root, extrude=.01)
    return root


def setup(key=.6, fill=.32, kick=1.):
    sc = setup_stage(samples=int(os.environ.get("SPP", 40)),
                     width=int(os.environ.get("W", 1920)), height=int(os.environ.get("H", 1080)))
    k, f = grade_stage(key, fill)
    kick_lights(.62 * kick)
    plexus_wall(.42 if os.environ.get("NO_PLEX") is None else 0)
    pal = palette()
    pal["wrong"] = mat("A Wrong", "#E8564A", .45, emit=1.3)
    pal["blank"] = mat("A Blank", "#3a4145", .6)
    pal["frost"] = mat("A Frost", "#b9d6e6", .22, emit=.25, clear=1.)
    pal["link"] = mat("A Link", "#33434e", .4, emit=.35)
    pal["wrap"] = mat("A Wrap", GREEN, .4, emit=.55)
    pal["hot"] = mat("A Hot", GREEN, .3, emit=4.)
    pal["beam"] = mat("A Beam", GREEN, .3, emit=2.2)
    pal["idb"] = mat("A IdBead", "#cfe3ee", .3, emit=1.6)
    pal["bub"] = mat("A Bubble", "#1a2732", .5, emit=.3)
    return sc, pal


def jolt(cam, frame, amp=.45):
    base = cam.o.location.copy()
    cam.at(frame, base, cam.tgt.location)
    cam.at(frame + 1, base + Vector((amp, 0, -amp)), cam.tgt.location)
    cam.at(frame + 3, base + Vector((-amp * .6, 0, amp * .5)), cam.tgt.location)
    cam.at(frame + 6, base, cam.tgt.location)


def hbar(x0, z, L, frac, m, track, a, d=20, h=2.4, y=-1., end=999999):
    tr = rbox("track", (L, .3, h), (x0 + L / 2, y + .4, z), track, bevel=.06)
    show_only(tr, a - 8, end)
    p = empty("barp", (x0, y, z))
    rbox("fill", (L, .5, h * .86), (L / 2, 0, 0), m, p, bevel=.06)
    show_tree(p, a, end)
    key(p, "scale", a - 1, Vector((.0005, 1, 1)))
    key(p, "scale", a, Vector((.0005, 1, 1)))
    key(p, "scale", a + d, Vector((frac, 1, 1)))
    return p


def show_tree(o, a, b):
    show_only(o, a, b)
    for c in o.children_recursive:
        show_only(c, a, b)


def stamp_sign(s, loc, pal, f, size=2.4, bg="steel", fg="white", tilt=-6, cam=None):
    st = sign(s, loc, pal, size=size, fg=fg, bg=bg, tilt=tilt)
    pop(st, f, 5)
    if cam:
        jolt(cam, f + 2, .3)
    return st


def optics(cam, f, lens=None, fstop=None):
    d = cam.o.data
    if lens:
        d.lens = lens
        d.keyframe_insert("lens", frame=f)
    if fstop:
        d.dof.aperture_fstop = fstop
        d.dof.keyframe_insert("aperture_fstop", frame=f)


def at(i, f=0.):
    return fr(TL[i]["start"] + f * TL[i]["dur"])


def wd(i, word, off=0.):
    return fr(word_t(i, word, off))


def grow(curve_obj, a, d=12):
    """Grow a curve tube along its length."""
    c = curve_obj.data
    c.bevel_factor_mapping_end = "SPLINE"
    c.bevel_factor_end = 0.
    c.keyframe_insert("bevel_factor_end", frame=1)
    c.keyframe_insert("bevel_factor_end", frame=a)
    c.bevel_factor_end = 1.
    c.keyframe_insert("bevel_factor_end", frame=a + d)


def orbit(cam, a, b, r, h, ang0, ang1, target=(0, 0, 4), lens=None, step=8):
    """Camera on a circular path around the array, keyed densely so it stays on the circle."""
    n = max(1, (b - a) // step)
    for k in range(n + 1):
        f = a + (b - a) * k // n
        u = k / n
        u = u * u * (3 - 2 * u)
        ang = math.radians(ang0 + (ang1 - ang0) * u)
        cam.at(f, (r * math.sin(ang), -r * math.cos(ang), h), target, lens=lens if k == 0 else None)



def along(pts, u):
    L = [0.]
    for p, q in zip(pts, pts[1:]):
        L.append(L[-1] + (q - p).length)
    t = u * L[-1]
    for k in range(len(pts) - 1):
        if L[k + 1] >= t:
            s = (t - L[k]) / max(1e-6, L[k + 1] - L[k])
            return pts[k].lerp(pts[k + 1], s)
    return pts[-1].copy()


_bead = {}


def bead_mesh(r):
    if r not in _bead:
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=9, radius=r)
        me = bpy.data.meshes.new(f"bead{r}")
        bm.to_mesh(me)
        me.materials.append(None)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        _bead[r] = me
    return _bead[r]



def thought(loc, pal, sx=4.2, sz=2.1, label=None, size=1.2):
    root = empty("thought", loc)
    uvs("tb", (0, 0, 0), (sx, .6, sz), pal["bub"], root)
    for k, (dx, dz, r) in enumerate(((-sx * .55, -sz * 1.2, .55), (-sx * .75, -sz * 1.75, .35))):
        uvs(f"tb{k}", (dx, 0, dz), (r, .4, r), pal["bub"], root, seg=12, ring=8)
    if label:
        text(label, (0, -.8, 0), size, pal["light"], root, extrude=.02)
    return root



# ---------------------------------------------------------------- a grid of monkeys
LZ = 1.0


def torus_mesh(R, minor):
    k = ("torus", R, minor)
    if k not in _bead:
        bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=minor, major_segments=40, minor_segments=8)
        o = bpy.context.object
        bpy.ops.object.shade_smooth()
        me = o.data
        me.materials.append(None)
        bpy.data.objects.remove(o)
        _bead[k] = me
    return _bead[k]


def inst(me, m, loc, parent=None, name="i"):
    o = bpy.data.objects.new(name, me)
    link(o, parent)
    o.location = loc
    o.material_slots[0].link = "OBJECT"
    o.material_slots[0].material = m
    return o


def disc_mesh(r, h):
    k = ("disc", r, h)
    if k not in _bead:
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=28, radius1=r, radius2=r, depth=h)
        me = bpy.data.meshes.new(f"disc{r}")
        bm.to_mesh(me)
        me.materials.append(None)
        bm.free()
        _bead[k] = me
    return _bead[k]


class Grid:
    """R x C monkeys on a torus, drawn flat. `shift` rolls the picture so a chosen agent sits in the middle (a torus has no
    edge, so any roll is the same network). `block` puts a gap after every block[0] rows and block[1] columns (arrays, or
    the tesseract's 4x4 blocks). One proto monkey is collection-instanced; `hero` is a full rig."""

    def __init__(self, pal, R, C, S=9., ms=.3, shift=(0, 0), block=None, gap=0., hero=None, pop_at=None, links_at=None,
                 wraps=True, wraps_at=None, link_mat="link", seed=3, link_r=.14, draw_links=True):
        self.pal, self.R, self.C, self.S, self.ms, self.shift, self.block, self.gap = pal, R, C, S, ms, shift, block, gap
        rnd = random.Random(seed)
        self.n = R * C
        self.w = self._axis(C - 1, 1) - self._axis(0, 1)
        self.h = self._axis(R - 1, 0) - self._axis(0, 0)
        before = set(bpy.data.objects)
        self.proto = Monkey("Proto", pal, (0, 0), ms)
        coll = bpy.data.collections.new("ProtoMonkey")
        bpy.context.scene.collection.children.link(coll)
        for o in [o for o in bpy.data.objects if o not in before]:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        bpy.context.view_layer.layer_collection.children["ProtoMonkey"].exclude = True
        self.hero_i = hero
        self.hero = Monkey("Pip", pal, tuple(self.pos(hero)[:2]), ms) if hero is not None else None
        self.inst = {}
        for i in range(self.n):
            if i == hero:
                continue
            e = bpy.data.objects.new(f"agent{i}", None)
            e.instance_type = "COLLECTION"
            e.instance_collection = coll
            link(e)
            e.location = self.pos(i)
            e.rotation_euler = (0, 0, math.radians(rnd.uniform(-14, 14)))
            self.inst[i] = e
            if pop_at is not None:
                d = self.pos(i).length / max(self.w, self.h)
                pop(e, pop_at + int(d * 26) + rnd.randint(0, 5), 7)
        self.link_objs = []
        if draw_links:
            for a, b in self.links():
                wrap = self.is_wrap(a, b)
                if wrap and not wraps:
                    continue
                t = tube(f"L{a}_{b}", self.path(a, b), link_r * (.55 if wrap else 1.), pal["wrap" if wrap else link_mat], order=3 if wrap else 2)
                self.link_objs.append((t, wrap))
                g = wraps_at if wrap else links_at
                if g is not None:
                    grow(t, g + rnd.randint(0, 10), 14)

    # geometry
    def _axis(self, k, ax):
        b = self.block[ax] if self.block else 0
        return k * self.S + (k // b) * self.gap if b else k * self.S

    def disp(self, i):
        r, c = divmod(i, self.C)
        return (r + self.shift[0]) % self.R, (c + self.shift[1]) % self.C

    def pos(self, i, z=0.):
        r, c = self.disp(i)
        x = self._axis(c, 1) - (self._axis(self.C - 1, 1)) / 2
        y = (self._axis(self.R - 1, 0)) / 2 - self._axis(r, 0)
        return Vector((x, y, z))

    def links(self):
        out = []
        for i in range(self.n):
            r, c = divmod(i, self.C)
            out.append((i, r * self.C + (c + 1) % self.C))
            out.append((i, ((r + 1) % self.R) * self.C + c))
        return out

    def is_wrap(self, a, b):
        (ra, ca), (rb, cb) = self.disp(a), self.disp(b)
        return abs(ra - rb) + abs(ca - cb) != 1

    def path(self, a, b, lift=0.):
        """Points from a to b along their link: straight, or an arc over the array when the link wraps round."""
        pa, pb = self.pos(a, LZ + lift), self.pos(b, LZ + lift)
        off = 5.6 * self.ms
        if not self.is_wrap(a, b):
            d = (pb - pa).normalized()
            return [pa + d * off + (pb - pa - d * 2 * off) * k / 8 for k in range(9)]
        horiz = abs(pa.x - pb.x) > abs(pa.y - pb.y)
        span = (self.w if horiz else self.h) / 2 + 2.2
        H = (.2 if horiz else .24) * 2 * span + 3
        sgn = 1 if (pa.x if horiz else pa.y) > 0 else -1
        pts = []
        for k in range(25):
            th = math.pi * k / 24
            u = sgn * span * math.cos(th)
            pts.append(Vector((u, pa.y, LZ + lift + H * math.sin(th))) if horiz else Vector((pa.x, u, LZ + lift + H * math.sin(th))))
        return pts

    # animation
    def node(self, i):
        return self.hero.root if i == self.hero_i else self.inst[i]

    def hop(self, i, f, h=2.):
        if i == self.hero_i:
            self.hero.hop_to(f, tuple(self.pos(i)[:2]), 8, h / self.ms * .6)
            return
        o, p = self.inst[i], self.pos(i)
        key(o, "location", f, p)
        key(o, "location", f + 4, p + Vector((0, 0, h)))
        key(o, "location", f + 8, p)

    def halo(self, i, f, m="hot", r=None, minor=None, dur=7, end=None):
        r = r or 6.2 * self.ms
        o = inst(torus_mesh(r, minor or r * .075), self.pal[m], self.pos(i, .32 + (.05 if m == "hot" else 0)), name="halo")
        pop(o, f, dur)
        if end:
            gone(o, end, 5)
        return o

    def lit(self, i, f, m="hot"):
        """Agent i holds the total: a glowing disc under it and a beam above."""
        r = 7.4 * self.ms
        for me, z in ((disc_mesh(round(r, 2), .16), .36), (disc_mesh(.16, 9.), 4.6 + 12 * self.ms)):
            o = inst(me, self.pal[m], self.pos(i, z), name="lit")
            pop(o, f, 6)

    def edge(self, a, b, f, m="beam", r=.3, dur=10, lift=.12, end=None):
        """A lit tube from a to b that grows from a's end at frame f."""
        t = tube(f"E{a}_{b}", self.path(a, b, lift), r * (.6 if self.is_wrap(a, b) else 1.), self.pal[m], order=3 if self.is_wrap(a, b) else 2)
        grow(t, f, dur)
        if end:
            show_only(t, 1, end)
        return t

    def bead(self, a, b, f0, dur, m="hot", size=.6, label=None, lift=.5):
        """A value travelling a -> b; with a label it carries a small card."""
        pts = self.path(a, b, lift)
        root = empty("bd", pts[0])
        o = inst(bead_mesh(round(size, 2)), self.pal[m], (0, 0, 0), root, "b")
        if label:
            c = card(label, self.pal, w=len(label) * .62 + 1.2, h=1.5, size=.78, loc=(0, -.6, size + 1.5), parent=root, bg="dark", fg="green")
        n = max(2, dur // 2)
        for k in range(n + 1):
            key(root, "location", f0 + dur * k // n, along(pts, k / n))
        show_tree(root, f0, f0 + dur + 1)
        return root


def big(s, loc, pal, size=3., fg="ink", bg="green", f=None, cam=None, tilt=0.):
    o = sign(s, loc, pal, size=size, fg=fg, bg=bg, tilt=tilt)
    if f is not None:
        pop(o, f, 7)
        if cam:
            jolt(cam, f + 2, .5)
    return o


def tk(t):
    return fr(TREE_TICKS[t])


def tree_scene(pal, G, tree, tick_fn, edge_m="beam", join_m="idb", beads=True, labels=(), until=None, lit=True, skip=(), no_wrap=False):
    """Play a real tree on grid G: the wave (tree edges grow parent -> child at the child's join tick), the sums (a bead child
    -> parent at the child's report tick, sized by how many processes it stands for), the total (halo + hop)."""
    for i, t in sorted(tree["join"].items(), key=lambda x: x[1]):
        f = tick_fn(t)
        if i in tree["parent"] and i not in skip and not (no_wrap and G.is_wrap(tree["parent"][i], i)):
            G.edge(tree["parent"][i], i, f - (tick_fn(t) - tick_fn(max(1, t - 1))) + 2, edge_m, dur=max(6, tick_fn(t) - tick_fn(max(1, t - 1)) - 2))
        G.halo(i, f, join_m, r=5.6 * G.ms, minor=.09)
    if beads:
        for i, (t, s, c) in tree["sub"].items():
            if i in skip or (until and t > until):
                continue
            f, d = tick_fn(t), max(8, tick_fn(t + 1) - tick_fn(t))
            G.bead(i, tree["parent"][i], f, d, "hot", .42 + .2 * c ** (1 / 3), label=f"{s:,} / {c}" if i in labels else None)
    if lit:
        for i, (t, s, c) in tree["total"].items():
            if until and t > until:
                continue
            f = tick_fn(t)
            G.lit(i, f)
            G.hop(i, f + (i % 3), 2.2)


# ================================================================ A. where we left off (title, a1-a3)
def s00():
    a, b = 1, cue("b1")
    sc, pal = setup(key=.55)
    G = Grid(pal, 4, 4, S=15., ms=.5, hero=1, link_mat="softglow", link_r=.28, wraps_at=at("a1", .2))
    m, hp = G.hero, G.pos(1)
    for i in range(16):
        G.halo(i, 1, "hot", r=3.4, minor=.24)
    cam = Camera(fstop=.8, lens=50)
    t1, t2, t3 = cue("a1"), cue("a2"), cue("a3")
    cam.at(a, hp + Vector((0, -46, 10)), hp + Vector((0, 0, 9)))
    cam.at(t1 + 6, hp + Vector((0, -44, 10)), hp + Vector((0, 0, 9)))
    orbit(cam, t1 + 20, t3 - 4, 100, 54, -8, 14, (0, 0, 10))
    cam.at(t3 + 4, (30, -110, 62), (0, 0, 10))
    cam.at(b, (60, -240, 190), (0, 0, 6))            # up and away: the grid is about to grow
    optics(cam, a, 50, .8)
    optics(cam, t1 + 6, 50, .8)
    optics(cam, t1 + 20, 40, 4.5)
    title = empty("title", hp + Vector((0, 3, 16)))
    text("PASS IT UP", (0, 0, 0), 4.2, pal["green"], title, extrude=.4)
    text("A VERY HONEST MONKEY FILM, PART TWO", (0, 0, -3.8), 1.05, pal["light"], title, extrude=.05, bold=False)
    pop(title, 6, 10)
    key(title, "location", at("a1", .08), title.location.copy())
    key(title, "location", at("a1", .08) + 12, title.location + Vector((0, 0, 60)))
    m.key(a, hop=50, squash=1.1, smile=.6, L=(-5.5, -.6, 9.5), R=(5.5, -.6, 9.5))
    m.key(9, hop=0, squash=.76)
    m.key(13, squash=1.05)
    m.key(17, squash=1., L=(-5, -.6, 5.2), R=(5, -.6, 5.2))
    m.key(t1 + 4, R=(6.5, -2.5, 11), smile=.9, head=(0, -4, 0))
    m.key(t1 + 22, R=(5, -.6, 5.2))
    s1 = big("6,637  EXACT  ·  16 OF 16", (0, 0, 31), pal, size=2.4, f=at("a1", .35))
    gone(s1, t2 - 1, 5)
    s2 = sign("QWEN3.8-27B  ×16", (0, 0, 25.5), pal, size=1.7, fg="light", bg="steel")
    pop(s2, wd("a1", "Each one"), 6)
    gone(s2, t2 - 1, 5)
    s3 = sign("FLOODING: EVERYONE COLLECTS EVERYONE", (0, 0, 31), pal, size=2.1, fg="green")
    pop(s3, t2 + 2, 6)
    gone(s3, t3 - 2, 4)
    rnd = random.Random(7)                           # illustrative flood packets
    for k in range(150):
        i = rnd.randrange(16)
        r, c = divmod(i, 4)
        j = rnd.choice([r * 4 + (c + 1) % 4, r * 4 + (c - 1) % 4, ((r + 1) % 4) * 4 + c, ((r - 1) % 4) * 4 + c])
        G.bead(i, j, t2 + rnd.randint(0, max(1, t3 - t2 + 10)), 14, rnd.choice(["hot", "idb"]), .55)
    s4 = sign("18 TICKS", (0, 0, 25.5), pal, size=1.7, fg="light", bg="steel")
    pop(s4, at("a2", .6), 6)
    gone(s4, t3 - 2, 4)
    s5 = big("SO WE MADE THE GRID BIGGER", (0, 0, 31), pal, size=2.4, fg="light", bg="steel", f=t3 + 1)
    gone(s5, b - 8, 5)
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


# ================================================================ B. the wall (b1-b4)
def s01():
    a, b = cue("b1"), cue("c1")
    sc, pal = setup(key=.5, fill=.3)
    G = Grid(pal, 10, 10, shift=(5, 5), hero=0, pop_at=a + 2, links_at=a + 14, wraps=False)
    m, hp = G.hero, G.pos(0)
    cam = Camera(fstop=5., lens=40)
    b2, b3, b4 = cue("b2"), cue("b3"), cue("b4")
    cam.at(a, (40, -230, 170), (0, 0, 6))
    orbit(cam, a + 16, b2 + 30, 136, 80, 8, -12, (0, 0, 6))
    orbit(cam, b2 + 31, b3 - 2, 136, 80, -12, -20, (0, 0, 6))
    # close on the hero's port for the bandwidth line
    cam.at(b3 + 14, hp + Vector((2.5, -22, 6.5)), hp + Vector((2.5, 0, 4.6)))
    cam.at(b4 - 4, hp + Vector((2., -21, 6.4)), hp + Vector((2.5, 0, 4.6)))
    cam.at(b4 + 12, (-16, -152, 88), (0, 0, 8))
    cam.at(b, (-20, -150, 86), (0, 0, 8))
    optics(cam, a, 40, 5.)
    optics(cam, b3 - 2, 40, 5.)
    optics(cam, b3 + 14, 50, 1.6)
    optics(cam, b4 - 4, 50, 1.6)
    optics(cam, b4 + 12, 40, 5.)
    rnd = random.Random(11)                          # illustrative flood packets, thinning out as the flood stalls
    span = b3 - a
    for k in range(520):
        i = rnd.randrange(100)
        r, c = divmod(i, 10)
        j = rnd.choice([r * 10 + (c + 1) % 10, r * 10 + (c - 1) % 10, ((r + 1) % 10) * 10 + c, ((r - 1) % 10) * 10 + c])
        if G.is_wrap(i, j):
            continue
        G.bead(i, j, a + 22 + int(span * rnd.random() ** 1.5), 13, rnd.choice(["hot", "idb", "idb"]), .34, lift=.3)
    s1 = sign("100 MONKEYS  ·  SAME RECIPE", (0, 0, 37.6), pal, size=4.48, fg="green")
    pop(s1, a + 8, 7)
    gone(s1, wd("b1", "Sixty") - 2, 4)
    s2 = big("TICK 60:  0 OF 100", (0, 0, 37.6), pal, size=5.32, fg="white", bg="steel", f=wd("b1", "not one"), cam=cam)
    gone(s2, b2 + 2, 4)
    s3 = sign("EACH HELD ABOUT 90 OF THE 100 NUMBERS", (0, 0, 37.6), pal, size=4.06, fg="light", bg="steel")
    pop(s3, b2 + 6, 6)
    gone(s3, b3 - 6, 4)
    for k, i in enumerate((23, 67, 81, 38, 92)):     # premature closure
        th = thought(G.pos(i) + Vector((3.4, 0, 9.5)), pal, sx=6.2, sz=1.9, label="\"NETWORK COMPLETE\"", size=.82)
        pop(th, wd("b2", "Several") + k * 3, 6)
        gone(th, b3 - 8, 4)
    # b3: one port, two slots
    slot = empty("slot", hp + Vector((2.5, -3., 7.6)))
    for k in range(2):
        c = empty(f"pc{k}", ((k - .5) * 4.3, 0, 0), slot)
        rbox("pp", (4.1, .2, .9), (0, 0, 0), pal["green"], c, bevel=.3)
        text(("8c7a7c7d:469", "e52ba3af:416")[k], (0, -.14, 0), .42, pal["ink"], c, extrude=.01)
    pop(slot, wd("b3", "A port"), 6)
    pl = sign("ONE PORT, ONE TICK: 2 NUMBERS", hp + Vector((2.5, -3., 9.1)), pal, size=.5, pad=.3)
    pop(pl, wd("b3", "A port") + 3, 6)
    pf = sign("PERFECT AGENTS: 50 TICKS TO FLOOD 100", hp + Vector((2.5, -3., 6.2)), pal, size=.5, pad=.3, fg="light", bg="steel")
    pop(pf, wd("b3", "Even perfect"), 6)
    for o in (slot, pl, pf):
        gone(o, b4 - 6, 4)
    m.key(a, smile=.2, brow=.2, head=(0, -4, 0))
    m.key(wd("b1", "not one"), smile=-.5, brow=-.6, squash=.95)
    m.key(b3 + 14, smile=-.2, brow=-.3, head=(14, -2, 0), R=(6.5, -3, 9.5))
    m.key(b4 + 8, R=(5, -.6, 5.2), head=(0, -4, 0))
    # b4: the prompt grows with the crowd
    bx = -46
    tl = sign("PROMPT, PER MONKEY, PER TICK", (0, -4, 40.4), pal, size=2.66, fg="light", bg="steel")
    pop(tl, b4 + 12, 6)
    for k, (lab, frac, mm) in enumerate((("TICK 1    1,300 TOKENS", 1300 / 5600, "green"), ("TICK 60  5,600 TOKENS", 1., "wrong"))):
        z = 41 - k * 4.6
        hbar(bx, z, 92, frac, pal[mm], pal["dark"], b4 + 16 + k * 10, d=18, h=3.2, y=-4)
        t = text(lab, (bx + 1.5, -5.4, z), 1.5, pal["ink" if k == 0 else "white"], extrude=.03, align="LEFT")
        pop(t, b4 + 20 + k * 10, 5)
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


# ================================================================ C + D. the tree, and the result (c1-d3)
def s02():
    a, b = cue("c1"), cue("e1")
    sc, pal = setup(key=.46, fill=.28)
    G = Grid(pal, 10, 10, shift=(5, 5), hero=0, wraps=False)
    T = T10["trees"]["A"]
    m, hp = G.hero, G.pos(0)
    cam = Camera(fstop=5., lens=40)
    c2, c3, c4, d1, d2, d3 = (cue(x) for x in ("c2", "c3", "c4", "d1", "d2", "d3"))
    cam.at(a, (-20, -150, 86), (0, 0, 8))
    cam.at(c2 - 8, (-8, -142, 82), (0, 0, 6))
    # in on the root for the first wave, then out with it
    cam.at(c2 + 6, hp + Vector((6, -58, 30)), hp + Vector((0, 4, 3)))
    cam.at(tk(3), hp + Vector((8, -64, 34)), hp + Vector((0, 4, 3)))
    orbit(cam, tk(5), c3 + 10, 134, 82, 4, 16, (0, 0, 5))
    orbit(cam, c3 + 11, c4 - 4, 134, 82, 16, 30, (0, 0, 5))
    # down low toward the root as the last four sums arrive
    cam.at(c4 + 10, hp + Vector((30, -70, 30)), hp + Vector((0, 2, 4)))
    cam.at(tk(23) + 4, hp + Vector((26, -74, 32)), hp + Vector((0, 2, 4)))
    orbit(cam, tk(24), d2 - 6, 140, 86, 18, -6, (0, 0, 6))
    cam.at(d2 + 12, (0, -152, 82), (0, 0, 12))
    cam.at(b, (3, -150, 80), (0, 0, 12))
    optics(cam, a, 40, 5.)
    optics(cam, c2 - 8, 40, 5.)
    optics(cam, c2 + 6, 46, 2.4)
    optics(cam, tk(3), 46, 2.4)
    optics(cam, tk(5), 40, 5.)
    optics(cam, c4 - 4, 40, 5.)
    optics(cam, c4 + 10, 46, 2.8)
    optics(cam, tk(23) + 4, 46, 2.8)
    optics(cam, tk(24), 40, 5.)
    tree_scene(pal, G, T, tk, labels=(1, 90, 9, 10))
    s0 = big("1982", (-22, 0, 38.5), pal, size=5.04, f=a + 4)
    s1 = sign("DON'T MAKE EVERYONE KNOW EVERYTHING", (12, 0, 38.5), pal, size=3.50, fg="light", bg="steel")
    pop(s1, wd("c1", "Don't"), 6)
    s2 = sign("ADD IT UP ALONG A TREE", (0, 0, 32.4), pal, size=4.20, fg="green")
    pop(s2, wd("c1", "Add it"), 7)
    for o in (s0, s1, s2):
        gone(o, c2 - 4, 4)
    w1 = sign("THE WAVE:  \"ARE YOU BELOW ME?\"", (0, 0, 38.5), pal, size=3.92, fg="light", bg="steel")
    pop(w1, tk(4), 6)
    gone(w1, c3 - 3, 4)
    w2 = sign("ONE MESSAGE UP:  A SUM AND A COUNT", (0, 0, 38.5), pal, size=3.92, fg="green")
    pop(w2, c3 + 2, 6)
    w3 = sign("THE HOST DOES THE ADDING", (0, 0, 32.9), pal, size=2.94, fg="light", bg="steel")
    pop(w3, wd("c3", "The host"), 6)
    for o in (w2, w3):
        gone(o, c4 - 2, 4)
    rt = empty("roottile", hp + Vector((0, -1, 7.4)))
    rbox("rp", (11.5, .3, 2.3), (0, 0, 0), pal["green"], rt, bevel=.12)
    text("49,689  /  100", (0, -.22, 0), 1.25, pal["ink"], rt, extrude=.03)
    pop(rt, tk(23) - 3, 7)
    gone(rt, tk(26), 5)
    w4 = sign("THE ROOT KNOWS IT IS COMPLETE", (0, 0, 38.5), pal, size=3.64, fg="green")
    pop(w4, wd("c4", "and knows"), 6)
    gone(w4, d1 + 2, 4)
    r1 = sign("QWEN3.8-27B  ·  SAME 100  ·  SAME SEED", (0, 0, 38.5), pal, size=3.36, fg="light", bg="steel")
    pop(r1, d1 + 6, 6)
    gone(r1, d2 - 2, 4)
    r2 = big("49,689  EXACT", (0, -4, 40.4), pal, size=6.16, f=d2 + 1, cam=cam)
    r3 = sign("100 OF 100  ·  0 WRONG", (0, -4, 33.1), pal, size=3.50)
    pop(r3, wd("d2", "No wrong"), 6)
    r4 = sign("PERFECT TREE: 32 TICKS  ·  THEY TOOK 33", (0, -4, 27.8), pal, size=2.94, fg="light", bg="steel")
    pop(r4, wd("d2", "A perfect"), 6)
    for o in (r2, r3, r4):
        gone(o, d3 - 2, 4)
    tl = sign("SECONDS A TICK", (-30, -4, 40.9), pal, size=2.38, fg="light", bg="steel")
    pop(tl, d3 + 2, 5)
    for k, (lab, frac, mm) in enumerate((("FLOOD  375", 1., "blank"), ("TREE    202", 202 / 375, "green"))):
        z = 42.5 - k * 4.4
        hbar(-48, z, 96, frac, pal[mm], pal["dark"], d3 + 5 + k * 6, d=14, h=3., y=-4)
        t = text(lab, (-46.5, -5.4, z), 1.45, pal["white" if k == 0 else "ink"], extrude=.03, align="LEFT")
        pop(t, d3 + 8 + k * 6, 5)
    m.key(a, smile=.3, head=(0, -4, 0))
    m.key(c2 + 2, R=(6.5, -2.5, 11.5), smile=.8, brow=.4)       # the root starts the wave
    m.key(tk(3), R=(5, -.6, 5.2))
    m.key(tk(22), smile=.6, brow=.3, head=(0, -4, 0))
    m.key(tk(23), L=(-5.5, -.6, 10.5), R=(5.5, -.6, 10.5), smile=1., brow=.6)
    m.key(tk(26), L=(-5, -.6, 5.2), R=(5, -.6, 5.2))
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


# ================================================================ E. the weakness (e1-e4): the real broken tree
LOST = [1, 61, 62, 63, 71, 72, 81, 82, 90, 91, 92]     # agent 63's branch in tree B of haiku-10x10-tree-verify-1110


def s03():
    a, b = cue("e1"), cue("g1")
    sc, pal = setup(key=.46, fill=.28)
    pal["treeb"] = mat("A TreeB", "#58b7ff", .35, emit=2.4)
    G = Grid(pal, 10, 10, shift=(0, 0), hero=55, wraps=False)       # tree B's root, agent 55, sits in the middle
    A, B = BROKEN["trees"]["A"], BROKEN["trees"]["B"]
    m, hp = G.hero, G.pos(55)
    e2, e3, e4 = cue("e2"), cue("e3"), cue("e4")
    cam = Camera(fstop=5., lens=40)
    cam.at(a, (3, -150, 80), (0, 0, 12))
    orbit(cam, a + 10, e2 + 10, 136, 82, 0, -14, (0, 0, 6))
    lp = sum((G.pos(i) for i in LOST), Vector()) / len(LOST)
    cam.at(wd("e2", "One monkey") + 14, lp + Vector((-14, -62, 44)), lp + Vector((0, 4, 2)))
    cam.at(wd("e2", "The total") - 6, lp + Vector((-10, -66, 46)), lp + Vector((0, 4, 2)))
    orbit(cam, wd("e2", "The total") + 10, e4 - 4, 140, 86, -10, 10, (0, 0, 7))
    cam.at(b, (8, -148, 82), (0, 0, 10))
    for f, ln, fs in ((a, 40, 5.), (e2 + 10, 40, 5.), (wd("e2", "One monkey") + 14, 44, 3.), (wd("e2", "The total") - 6, 44, 3.),
                      (wd("e2", "The total") + 10, 40, 5.)):
        optics(cam, f, ln, fs)
    grown = a - 40                                                   # tree B stands from the first frame
    for i, p in B["parent"].items():
        lost = i in LOST
        G.edge(p, i, grown, "beam", end=wd("e2", "eleven") + 4 if lost else None)
    fall = wd("e2", "eleven")
    for i in LOST:                                                   # the branch goes red, then dark
        par = B["parent"][i]
        t = G.edge(par, i, grown, "wrong", r=.34)
        show_only(t, wd("e2", "One monkey") + 4, 999999)
        gone(t, fall + 26 + (i % 5), 8)
        h = G.halo(i, wd("e2", "One monkey") + 4 + (i % 4), "wrong", r=5.8 * G.ms, minor=.2)
        gone(h, fall + 26 + (i % 5), 8)
    s1 = sign("ONE SLIP, AND A BRANCH VANISHES", (0, 0, 38.5), pal, size=3.9, fg="light", bg="steel")
    pop(s1, wd("e1", "One slip"), 6)
    gone(s1, e2 - 2, 4)
    s2 = sign("CLAUDE HAIKU RUN  ·  SECOND TREE", (0, 0, 38.5), pal, size=3.1, fg="light", bg="steel")
    pop(s2, e2 + 3, 6)
    gone(s2, wd("e2", "One monkey") + 8, 4)
    s3 = big("11 MONKEYS GONE", (lp.x, lp.y - 2, 20), pal, size=2.2, fg="white", bg="wrong", f=fall, cam=cam)
    gone(s3, wd("e2", "The total") + 6, 4)
    s4 = sign("44,294  OVER  89", (-26, 0, 38.5), pal, size=3.6, fg="wrong")
    pop(s4, wd("e2", "The total") + 12, 6)
    s5 = sign("TRUE:  49,689  OVER  100", (30, 0, 38.5), pal, size=3.2, fg="green")
    pop(s5, wd("e2", "It was wrong"), 6)
    for o in (s4, s5):
        gone(o, e3 - 2, 4)
    # e3: a second tree from another root
    ra = G.pos(0)
    span = max(1, wd("e3", "and trust") - e3 - 6)
    last = max(A["join"].values())
    for i, p in A["parent"].items():
        G.edge(p, i, e3 + 4 + int(span * (A["join"][i] - 1) / last), "treeb", r=.2, lift=.9, dur=8)
    G.halo(0, e3 + 2, "treeb", r=8 * G.ms, minor=.3)
    G.halo(55, e3 + 2, "hot", r=8 * G.ms, minor=.3)
    t1 = sign("TREE A:  49,689 / 100", (-30, 0, 38.5), pal, size=3.2, fg="light", bg="steel")
    t2 = sign("TREE B:  44,294 / 89", (30, 0, 38.5), pal, size=3.2, fg="wrong")
    pop(t1, wd("e3", "two trees") + 6, 6)
    pop(t2, wd("e3", "two roots") + 4, 6)
    t3 = big("THEY DISAGREE:  DON'T TRUST IT", (0, 0, 31.0), pal, size=3.4, fg="white", bg="steel", f=wd("e3", "The second"), cam=cam)
    for o in (t1, t2, t3):
        gone(o, e4 - 2, 4)
    r1 = sign("THE HOST REFUSES A WAVE ON THE PARENT'S PORT", (0, 0, 38.5), pal, size=3., fg="green")
    pop(r1, e4 + 2, 6)
    r2 = big("RERUN:  100 OF 100  ·  BOTH TREES AGREE", (0, 0, 31.0), pal, size=3.4, f=wd("e4", "The rerun"), cam=cam)
    for i in range(100):                                             # the rerun: everyone lit
        G.lit(i, wd("e4", "The rerun") + 2 + int(G.pos(i).length / 6))
    m.key(a, smile=.2, head=(0, -4, 0))
    m.key(fall, smile=-.6, brow=-.7, squash=.94)
    m.key(e3 + 6, smile=.3, brow=.2)
    m.key(wd("e4", "The rerun") + 4, L=(-5.5, -.6, 10.5), R=(5.5, -.6, 10.5), smile=1., brow=.6)
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


# ================================================================ G. the swarm (g1): 112 Haiku agents, 7 arrays, slow seams
def s04():
    a, b = cue("g1"), cue("f1")
    sc, pal = setup(key=.5, fill=.3)
    T = _load("tree4x28")["trees"]["A"]
    G = Grid(pal, 4, 28, S=7.4, ms=.26, shift=(2, 14), block=(4, 4), gap=5.5, hero=0, wraps=False, pop_at=a + 2, links_at=a + 10)
    for (x, y), wrap in zip(G.links(), [G.is_wrap(*l) for l in G.links()]):      # seams between arrays: gold
        if not wrap and abs(G.pos(x).x - G.pos(y).x) > G.S + 1:
            tube(f"seam{x}", G.path(x, y, .2), .22, pal["gold"])
    m = G.hero
    cam = Camera(fstop=5., lens=36)
    cam.at(a, (0, -52, 34), (0, 6, 2))                 # in close on the root's array, then out as the tree spreads along the strip
    cam.at(a + 24, (0, -56, 36), (0, 6, 2))
    cam.at(wd("g1", "all exact") - 10, (0, -176, 112), (0, 16, 0))
    cam.at(b, (0, -184, 116), (0, 16, 0))
    span = wd("g1", "all exact") - (a + 30)
    f_of = lambda t: a + 30 + int(span * t / 64)
    tree_scene(pal, G, T, f_of, beads=False, lit=True, no_wrap=True)
    s1 = sign("112 CLAUDE HAIKU AGENTS  ·  7 ARRAYS  ·  SLOW SEAMS", (0, 30, 30), pal, size=4.2, fg="light", bg="steel")
    pop(s1, a + 8, 6)
    s2 = big("112 OF 112 EXACT", (-44, 30, 19), pal, size=4.6, f=wd("g1", "all exact"), cam=cam)
    s3 = sign("HALF THE TOKENS OF FLOODING", (50, 30, 19), pal, size=3.6)
    pop(s3, wd("g1", "for half"), 6)
    m.key(a, smile=.6, head=(0, -4, 0))
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


# ================================================================ F. the tesseract (f1-f6): the ring sum, and how it broke
def tpos(G, co):
    a4, b4, c4, d4 = co
    return G.pos((a4 * 4 + b4) * 16 + c4 * 4 + d4)


def ring_pts(G, co, dim, z=1.2):
    """The closed path through the four monkeys of one ring (dimension dim), as arcs hopping from each to the next."""
    pts = []
    for k in range(4):
        c0, c1 = list(co), list(co)
        c0[dim], c1[dim] = k, (k + 1) % 4
        p, q = tpos(G, c0), tpos(G, c1)
        H = (.12 if k < 3 else .2) * (q - p).length + .8
        for j in range(12):
            u = j / 12
            pts.append(Vector((p.x + (q.x - p.x) * u, p.y + (q.y - p.y) * u, z + H * math.sin(math.pi * u))))
    return pts + [pts[0]]


def s05():
    a, b = cue("f1"), cue("h1")
    sc, pal = setup(key=.46, fill=.28)
    G = Grid(pal, 16, 16, S=5.6, ms=.19, block=(4, 4), gap=7., hero=None, draw_links=False, pop_at=a + 2)
    f2, f3, f4, f5, f6 = (cue(x) for x in ("f2", "f3", "f4", "f5", "f6"))
    cam = Camera(fstop=5., lens=38)
    cam.at(a, (24, -140, 124), (0, 0, 2))
    orbit(cam, a + 14, f2 - 2, 146, 108, 10, -8, (0, 0, 2))
    orbit(cam, f2 - 1, f4 - 2, 146, 108, -8, 14, (0, 0, 2))
    orbit(cam, f4 - 1, b, 152, 112, 14, -4, (0, 0, 4))
    optics(cam, a, 38, 5.)
    s1 = sign("256 MONKEYS  ·  A TESSERACT  ·  4 RINGS EACH", (0, 0, 46), pal, size=4.42, fg="green")
    pop(s1, a + 8, 6)
    gone(s1, f2 - 2, 4)
    # four phases, one dimension each: within a block (columns, then rows), then across blocks
    order = (3, 1, 2, 0)                                             # drawn small rings first so the picture reads
    names = ("RING 1 OF 4:  ×4", "RING 2 OF 4:  ×16", "RING 3 OF 4:  ×64", "RING 4 OF 4:  ×256")
    t0, t1 = f2 + 6, f3 - 2
    per = (t1 - t0) // 4
    rnd = random.Random(5)
    for ph, dim in enumerate(order):
        f = t0 + ph * per
        lab = sign(names[ph], (0, 0, 46), pal, size=4.16, fg="light", bg="steel")
        pop(lab, f, 5)
        gone(lab, f + per - 3, 3)
        seen = set()
        for i in range(256):
            co = [i // 64, i // 16 % 4, i // 4 % 4, i % 4]
            co[dim] = 0
            if tuple(co) in seen:
                continue
            seen.add(tuple(co))
            pts = ring_pts(G, co, dim)
            t = tube(f"R{ph}_{i}", pts, .1 if dim in (1, 3) else .075, pal["beam"], order=3)
            grow(t, f + rnd.randint(0, 4), max(8, per - 10))
            show_only(t, f, f + per + 2)
            if len(seen) % (1 if dim in (1, 3) else 1) == 0:
                root = empty("rb", pts[0])
                inst(bead_mesh(.5), pal["hot"], (0, 0, 0), root, "b")
                n = 12
                for k in range(n + 1):
                    key(root, "location", f + 4 + (per - 6) * k // n, along(pts, k / n))
                show_tree(root, f + 4, f + per)
    s3 = sign("PERFECT AGENTS: 17 TICKS", (-34, 0, 46), pal, size=3.90, fg="green")
    s4 = sign("FLOODING: 128 TICKS", (36, 0, 46), pal, size=3.90, fg="light", bg="steel")
    pop(s3, f3 + 2, 6)
    pop(s4, wd("f3", "Flooding"), 6)
    for o in (s3, s4):
        gone(o, f4 - 2, 4)
    # f4: it broke. the real split: 186 exact, 70 wrong (56 of them the same doubled number)
    brk = big("IT BROKE", (0, 0, 46), pal, size=6.50, fg="white", bg="wrong", f=f4 + 1, cam=cam)
    gone(brk, wd("f4", "and every") + 4, 4)
    bad = 82                                                          # the agent whose number was counted twice
    order_i = sorted(range(256), key=lambda i: (G.pos(i) - G.pos(bad)).length)
    wrong = set(order_i[:70])
    h0 = G.halo(bad, wd("f4", "One monkey"), "wrong", r=2.4, minor=.5)
    s5 = sign("ONE NUMBER, COUNTED TWICE", (0, 0, 46), pal, size=4.16, fg="wrong")
    pop(s5, wd("f4", "counted twice") - 4, 6)
    gone(s5, f5 - 2, 4)
    spread = f5 - wd("f4", "and every")
    for k, i in enumerate(order_i):
        f = wd("f4", "and every") + int(spread * k / 256)
        if i in wrong:
            G.halo(i, f, "wrong", r=1.5, minor=.34)
        else:
            G.lit(i, f + 6)
    r1 = sign("CLAUDE HAIKU:  186 OF 256 EXACT", (-36, 0, 46), pal, size=3.77)
    r2 = sign("70 CONFIDENTLY WRONG", (40, 0, 46), pal, size=3.77, fg="wrong")
    r3 = sign("QWEN3.8-27B:  FAILED WITH ONLY 16", (0, 0, 38.5), pal, size=3.38, fg="light", bg="steel")
    pop(r1, f5 + 2, 6)
    pop(r2, wd("f5", "Seventy"), 6)
    pop(r3, wd("f5", "And the"), 6)
    for o in (r1, r2, r3):
        gone(o, f6 - 2, 4)
    n1 = sign("NO SECOND TREE TO CATCH A SLIP", (0, 0, 46), pal, size=4.16, fg="light", bg="steel")
    pop(n1, f6 + 2, 6)
    n2 = big("NEXT: THE HOST CHECKS EVERY WRITE", (0, 0, 38), pal, size=4.16, f=wd("f6", "So next"))
    n3 = big("TOMORROW", (0, -6, 29.5), pal, size=5.46, fg="white", bg="steel", f=wd("f6", "That's tomorrow"), cam=cam)
    return a, b


# ================================================================ H. the lesson (h1-h2)
def s06():
    a, b = cue("h1"), fr(TL["_total"])
    sc, pal = setup(key=.5)
    G = Grid(pal, 4, 4, S=15., ms=.5, hero=1, link_mat="softglow", link_r=.28, wraps=False)
    m, hp = G.hero, G.pos(1)
    for i in range(16):
        G.halo(i, 1, "hot", r=3.4, minor=.24)
    h2 = cue("h2")
    cam = Camera(fstop=4., lens=40)
    cam.at(a, (0, -96, 50), (0, 0, 12))
    cam.at(h2 - 4, (0, -90, 47), (0, 0, 12))
    cam.at(h2 + 10, hp + Vector((0, -66, 9)), hp + Vector((0.8, 0, 4.4)))
    cam.at(b, hp + Vector((0, -62, 9)), hp + Vector((0.8, 0, 4.4)))
    optics(cam, h2 - 4, 40, 4.)
    optics(cam, h2 + 10, 46, 1.6)
    f1 = sign("NOBODY HAS TO KNOW EVERYTHING.", (0, 0, 33), pal, size=2.5, fg="green")
    pop(f1, a + 1, 7)
    f2 = sign("BUT SOMEBODY HAS TO CHECK.", (0, 0, 26.5), pal, size=2.5, fg="light", bg="steel")
    pop(f2, wd("h1", "But somebody"), 7)
    for o in (f1, f2):
        gone(o, h2 - 2, 4)
    m.key(a, smile=.8, head=(0, -6, 0))
    for k, (dx, dz) in enumerate(((7, 12.5), (7.6, 12.2), (6.6, 12.8), (7.6, 12.2), (6.8, 12.6))):
        m.key(h2 + 6 + k * 6, R=(dx, -2.5, dz), smile=1., brow=.5, head=(0, -6, 0))
    m.hold(b)
    m.idle(a, b)
    m.finish()
    return a, b


SHOTS = dict(s00=s00, s01=s01, s02=s02, s03=s03, s04=s04, s05=s05, s06=s06)
ORDER = [("s00", "title"), ("s01", "b1"), ("s02", "c1"), ("s03", "e1"), ("s04", "g1"), ("s05", "f1"), ("s06", "h1")]
