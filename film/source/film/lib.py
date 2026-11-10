"""Blender helpers for the Chemistry monkey film.

The look comes from the Atlas 3DSS brand guide: the official render stage, the
Atlas Model Blue Gray material, green kick lights, and #82C232 / #686868 /
#556d79 / #E0E0E0. All on-screen numbers are real geometry set in Value Sans.
Frame numbers are global film frames at 24 fps, so cues come from timeline.json.
"""
import bpy, bmesh, json, math, os
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 24
TL = json.load(open(f"{ROOT}/timeline.json"))
GREEN, GREY, STEEL, LIGHT = "#82C232", "#686868", "#556d79", "#E0E0E0"


def fr(t):
    return 1 + round(t * FPS)


def cue(i, off=0.0):
    return fr(TL[i]["start"] + off)


def cue_end(i, off=0.0):
    return fr(TL[i]["end"] + off)


def lin(h):
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4)
    return (*out, 1.0)


_mats = {}


def mat(name, color=None, rough=.45, metal=0., emit=0., spec=.3, clear=0.):
    key = name
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    if color:
        p.inputs["Base Color"].default_value = lin(color) if isinstance(color, str) else color
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Specular IOR Level"].default_value = spec
    p.inputs["Coat Weight"].default_value = clear
    if emit:
        p.inputs["Emission Color"].default_value = lin(color) if isinstance(color, str) else color
        p.inputs["Emission Strength"].default_value = emit
    _mats[key] = m
    return m


def resin():
    return bpy.data.materials["Atlas - Model Blue Gray"]


def palette():
    return dict(
        resin=resin(),
        green=mat("A Green", GREEN, .38, clear=.25),
        glow=mat("A Glow", GREEN, .4, emit=1.9),
        softglow=mat("A SoftGlow", GREEN, .4, emit=1.4),
        grey=mat("A Grey", GREY, .6),
        dark=mat("A Dark", "#17191b", .5),
        steel=mat("A Steel", STEEL, .5),
        light=mat("A Light", LIGHT, .62),
        white=mat("A White", "#f3f5f4", .5, emit=.25),
        ink=mat("A Ink", "#14171a", .55),
        eye=mat("A Eye", "#050607", .08, spec=.8, clear=1.),
        gold=mat("A Gold", "#E9B949", .3, emit=.9, clear=1.),
        red=mat("A Warn", "#E0E0E0", .4, emit=2.2),
    )


# ---------------------------------------------------------------- stage
def setup_stage(samples=48, width=1920, height=1080):
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.compression = 15
    sc.render.film_transparent = False
    sc.render.use_persistent_data = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = .03
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_use_gpu = True
    sc.cycles.max_bounces = 6
    sc.cycles.glossy_bounces = 3
    sc.cycles.transmission_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    for c in bpy.data.collections:
        if c.name.startswith("Material Cubes"):
            lc = bpy.context.view_layer.layer_collection.children.get(c.name)
            if lc:
                lc.exclude = False
            for o in c.objects:
                o.hide_render = True
                o.hide_viewport = False
    o = bpy.data.objects.get("Cube")
    if o:
        o.hide_render = True
    enable_gpu()
    return sc


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                break
        except Exception:
            pass
    for d in prefs.devices:
        d.use = d.type != "CPU"


def area(name, loc, target=(0, 0, 8), size=30, energy=1000, color=GREEN):
    l = bpy.data.lights.new(name, "AREA")
    l.size = size
    l.energy = energy
    l.color = lin(color)[:3]
    o = bpy.data.objects.new(name, l)
    bpy.context.scene.collection.objects.link(o)
    o.visible_glossy = False  # the mirror floor must not show the light itself
    o.visible_camera = False
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def grade_stage(key=.6, fill=.32):
    """Darker and cooler than the stage default: the guide asks for cool feel and contrast."""
    sc = bpy.context.scene
    for n, k in (("Area Llght Right - Key", key), ("Area Liight Left - Fill", fill)):
        l = bpy.data.objects[n].data
        l.energy *= k
        l.color = (.86, .93, 1.)
    for look in ("Filmic - Medium High Contrast", "Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            pass
    fl = bpy.data.materials["Black Matte Backdrop"].node_tree.nodes["Principled BSDF"]
    fl.inputs["Base Color"].default_value = (.006, .007, .009, 1)
    fl.inputs["Roughness"].default_value = .14
    fl.inputs["Specular IOR Level"].default_value = .38
    w = sc.world.node_tree.nodes.get("Background")
    if w:
        w.inputs["Strength"].default_value *= .12
    return bpy.data.objects["Area Llght Right - Key"], bpy.data.objects["Area Liight Left - Fill"]


def plexus_wall(strength=.5, y=95):
    """The official Atlas plexus loop, far behind the set on a wall the camera defocuses."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, y, 46), rotation=(math.pi / 2, 0, 0))
    p = bpy.context.object
    p.name = "PlexusWall"
    p.scale = (340, 191, 1)
    m = bpy.data.materials.new("plexus")
    m.use_nodes = True
    nt = m.node_tree
    im = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(f"{ROOT}/brand/Atlas Plexus Background Loop.mp4")
    im.image = img
    im.image_user.frame_duration = img.frame_duration
    im.image_user.use_cyclic = True
    im.image_user.use_auto_refresh = True
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(im.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    p.data.materials.append(m)
    p.visible_shadow = False
    p.visible_diffuse = False
    return em


def light_key(o, frame, energy=None, color=None):
    if energy is not None:
        o.data.energy = energy
    if color is not None:
        o.data.color = color
    o.data.keyframe_insert("energy", frame=frame)
    o.data.keyframe_insert("color", frame=frame)


def kick_lights(strength=.62):
    """The guide's green kick lights: rim from behind each side. The floor is excluded so it stays black."""
    lights = [area("KickL", (-55, 32, 22), energy=52000 * strength, size=26),
              area("KickR", (58, 30, 34), energy=36000 * strength, size=22)]
    floor = bpy.data.objects["Mirror Floor and Background"]
    coll = bpy.data.collections.new("KickExclude")
    coll.objects.link(floor)
    for l in lights:
        l.light_linking.receiver_collection = coll
    coll.collection_objects[0].light_linking.link_state = "EXCLUDE"
    return lights


# ---------------------------------------------------------------- mesh helpers
def link(o, parent=None):
    bpy.context.scene.collection.objects.link(o)
    if parent:
        o.parent = parent
    return o


def empty(name="e", loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None)
    o.location = loc
    return link(o, parent)


def uvs(name, loc, scale, m=None, parent=None, seg=32, ring=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=ring, radius=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale if hasattr(scale, "__len__") else (scale,) * 3
    bpy.ops.object.shade_smooth()
    if m:
        o.data.materials.append(m)
    if parent:
        o.parent = parent
    return o


def rbox(name, size, loc=(0, 0, 0), m=None, parent=None, bevel=.12):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True, location=False, rotation=False)
    if bevel:
        b = o.modifiers.new("b", "BEVEL")
        b.width = bevel
        b.segments = 3
        b.limit_method = "ANGLE"
        smooth_angle(o)
    if m:
        o.data.materials.append(m)
    if parent:
        o.parent = parent
    return o


def smooth_angle(o, deg=40):
    """Blender 4.5: per-angle smoothing is a mesh attribute set by this operator."""
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(deg))


def cyl(name, r, h, loc=(0, 0, 0), m=None, parent=None, bevel=.1, seg=64, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=seg, radius=r, depth=h, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    if bevel:
        b = o.modifiers.new("b", "BEVEL")
        b.width = bevel
        b.segments = 3
        b.limit_method = "ANGLE"
    smooth_angle(o)
    if m:
        o.data.materials.append(m)
    if parent:
        o.parent = parent
    return o


def fuse(name, parts, voxel=.11, m=None, smooth=6):
    """Join parts and voxel-remesh them into one sculpted-looking resin casting."""
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        bpy.context.view_layer.objects.active = p
        p.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    r = o.modifiers.new("r", "REMESH")
    r.mode = "VOXEL"
    r.voxel_size = voxel
    r.use_smooth_shade = True
    s = o.modifiers.new("s", "SMOOTH")
    s.iterations = smooth
    s.factor = .9
    bpy.ops.object.modifier_apply(modifier="r")
    bpy.ops.object.modifier_apply(modifier="s")
    o.data.materials.clear()
    if m:
        o.data.materials.append(m)
    return o


def tube(name, pts, radius, m, parent=None, order=3):
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.bevel_depth = radius
    c.bevel_resolution = 5
    c.resolution_u = 14
    c.use_fill_caps = True
    s = c.splines.new("NURBS")
    s.points.add(len(pts) - 1)
    for p, v in zip(s.points, pts):
        p.co = (*v, 1)
    s.use_endpoint_u = True
    s.order_u = min(order, len(pts))
    c.materials.append(m)
    o = bpy.data.objects.new(name, c)
    return link(o, parent)


_fonts = {}


def font(bold=True):
    k = "value-sans-bold.ttf" if bold else "value-sans-regular.ttf"
    if k not in _fonts:
        _fonts[k] = bpy.data.fonts.load(f"{ROOT}/fonts/{k}")
    return _fonts[k]


def text(s, loc=(0, 0, 0), size=3., m=None, parent=None, extrude=.12, align="CENTER", rot=(math.pi / 2, 0, 0), bold=True, name=None, valign="CENTER"):
    c = bpy.data.curves.new(name or f"t_{s[:10]}", "FONT")
    c.body = s
    c.font = font(bold)
    c.size = size
    c.align_x = align
    c.align_y = valign
    c.extrude = extrude
    c.bevel_depth = min(.03, extrude * .25) if extrude else 0
    c.resolution_u = 6
    if m:
        c.materials.append(m)
    o = bpy.data.objects.new(c.name, c)
    o.location = loc
    o.rotation_euler = rot
    return link(o, parent)


# ---------------------------------------------------------------- animation
def key(o, attr, frame, val=None, interp=None):
    if val is not None:
        setattr(o, attr, val)
    o.keyframe_insert(attr, frame=frame)
    if interp and o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            if fc.data_path == attr:
                for kp in fc.keyframe_points:
                    if int(kp.co.x) == int(frame):
                        kp.interpolation = interp


def pop(o, frame, dur=8, scale=None, hold_before=True):
    """Scale from nothing to full with overshoot, the stop-motion 'appear'."""
    s = Vector(scale if scale is not None else o.scale)
    if hold_before and frame > 1:
        key(o, "scale", frame - 1, s * 0.0001)
    key(o, "scale", frame, s * 0.02)
    key(o, "scale", frame + int(dur * .65), s * 1.12)
    key(o, "scale", frame + dur, s)


def gone(o, frame, dur=6):
    s = Vector(o.scale)
    key(o, "scale", frame, s)
    key(o, "scale", frame + dur, s * .0001)


def show_only(o, a, b):
    """Visible in frames [a, b); hidden before and after, with no fade."""
    for f, h in ((1, True), (a, False), (b, True)):
        if f < 1:
            continue
        o.hide_render = h
        o.hide_viewport = h
        o.keyframe_insert("hide_render", frame=f)
        o.keyframe_insert("hide_viewport", frame=f)
    if o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            if fc.data_path.startswith("hide"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


class Camera:
    def __init__(self, fstop=3.2, lens=50):
        self.tgt = empty("camTarget", (0, 0, 8))
        cd = bpy.data.cameras.new("FilmCam")
        cd.lens = lens
        cd.sensor_width = 36
        cd.clip_start = .5
        cd.clip_end = 8000
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.aperture_blades = 7
        cd.dof.focus_object = self.tgt
        self.o = link(bpy.data.objects.new("FilmCam", cd))
        t = self.o.constraints.new("TRACK_TO")
        t.target = self.tgt
        t.track_axis = "TRACK_NEGATIVE_Z"
        t.up_axis = "UP_Y"
        bpy.context.scene.camera = self.o
        for m in list(bpy.context.scene.timeline_markers):
            bpy.context.scene.timeline_markers.remove(m)

    def at(self, frame, loc, target, lens=None, fstop=None):
        key(self.o, "location", frame, Vector(loc))
        key(self.tgt, "location", frame, Vector(target))
        if lens:
            self.o.data.lens = lens
            self.o.data.keyframe_insert("lens", frame=frame)
        if fstop:
            self.o.data.dof.aperture_fstop = fstop
            self.o.data.dof.keyframe_insert("aperture_fstop", frame=frame)


# ---------------------------------------------------------------- banana
def banana_mesh(name="banana", L=3.9, r=.46):
    bm = bmesh.new()
    rings, sides, arc = 30, 12, math.radians(88)
    R = L / arc
    verts = []
    for i in range(rings + 1):
        u = i / rings
        a = (u - .5) * arc
        ctr = Vector((math.sin(a) * R, 0, (1 - math.cos(a)) * R))
        tang = Vector((math.cos(a), 0, math.sin(a)))
        nrm = Vector((-math.sin(a), 0, math.cos(a)))
        bin_ = Vector((0, 1, 0))
        # Full through the middle, a blunt tip at one end and a narrow stem at the other.
        rad = r * min(1.0, (u / .16) ** .55) * min(1.0, ((1 - u) / .2) ** .8)
        if u > .84:
            rad = max(rad, r * .3)  # stem
        if u == 1.0 or u == 0.0:
            rad = r * .12
        ringv = []
        for j in range(sides):
            t = j / sides * 2 * math.pi
            ridge = 1 + .045 * math.cos(5 * t)
            ringv.append(bm.verts.new(ctr + (nrm * math.cos(t) + bin_ * math.sin(t)) * rad * ridge))
        verts.append(ringv)
    for i in range(rings):
        for j in range(sides):
            bm.faces.new((verts[i][j], verts[i][(j + 1) % sides], verts[i + 1][(j + 1) % sides], verts[i + 1][j]))
    bm.faces.new(verts[0][::-1])
    bm.faces.new(verts[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


class BananaGrid:
    """One banana per task, in real task order. Correct = Atlas green; wrong = dark grey."""

    def __init__(self, wrong, origin, pal, cols=10, dx=3.95, dz=3.55, scale=1., mesh=None, parent=None, prefix="", panel=True):
        self.items = []
        self.origin = Vector(origin)
        self.mesh = mesh or banana_mesh()
        if panel:
            w, h = cols * dx * scale + 1.2 * scale, 4 * dz * scale + .6 * scale
            pc = self.origin + Vector(((cols - 1) * dx * scale / 2, 1.6, -1.5 * dz * scale))
            self.panel = rbox(f"{prefix}panel", (w, .6, h), pc, mat("gridpanel", "#222b32", .7), parent, bevel=.18)
        for i in range(40):
            c, row = i % cols, i // cols
            loc = self.origin + Vector((c * dx * scale, 0, -row * dz * scale))
            o = bpy.data.objects.new(f"{prefix}b{i + 1:02d}", self.mesh)
            link(o, parent)
            o.location = loc
            o.rotation_euler = (math.radians(10), math.radians(32), 0)
            o.scale = (scale,) * 3
            # Material lives on the object slot so the same mesh can be green or grey.
            if len(o.material_slots) == 0:
                o.data.materials.append(pal["green"])
            o.material_slots[0].link = "OBJECT"
            o.material_slots[0].material = pal["grey" if (i + 1) in wrong else "green"]
            self.items.append(o)

    def reveal(self, start, step=1.0, dur=6):
        for i, o in enumerate(self.items):
            pop(o, int(start + i * step), dur)

    def center(self, i):
        return self.items[i - 1].location.copy()


# ---------------------------------------------------------------- monkey
class Monkey:
    """A tabletop-miniature monkey: resin casting on a round base, so it hops rather than walks."""

    def __init__(self, name, pal, loc=(0, 0), scale=1., glasses=True, visor=False, nightcap=False, base_mat=None):
        self.pal, self.name = pal, name
        self.state = dict(pos=(loc[0], loc[1]), turn=0., hop=0., lean=(0., 0.), squash=1.,
                          head=(0., 0., 0.), L=(-5.2, -.6, 6.2), R=(5.2, -.6, 6.2), brow=0., smile=.6, open=0., blink=0.,
                          tip=0.)
        self.root = empty(f"{name}_root", (loc[0], loc[1], 0))
        self.root.scale = (scale,) * 3
        R = resin()
        self.base = cyl(f"{name}_base", 5.2, .9, (0, 0, .45), base_mat or pal["dark"], self.root, bevel=.16)
        cyl(f"{name}_rim", 5.22, .14, (0, 0, .7), pal["green"], self.root, bevel=.03)
        self.body = empty(f"{name}_body", (0, 0, .9), self.root)
        # Feet, legs and torso are one casting.
        parts = [uvs("t", (0, 0, 5.2), (3.05, 2.55, 3.5), seg=24, ring=16),
                 uvs("t2", (0, -.7, 4.6), (2.45, 2.05, 2.75), seg=24, ring=16),
                 uvs("sh", (0, 0, 7.7), (2.55, 2.05, 1.5), seg=24, ring=16),
                 uvs("legL", (-1.7, -.2, 1.9), (1.2, 1.25, 1.9), seg=20, ring=12),
                 uvs("legR", (1.7, -.2, 1.9), (1.2, 1.25, 1.9), seg=20, ring=12),
                 uvs("fL", (-2.0, -1.5, .55), (1.3, 2.0, .6), seg=20, ring=12),
                 uvs("fR", (2.0, -1.5, .55), (1.3, 2.0, .6), seg=20, ring=12)]
        self.torso = fuse(f"{name}_torso", parts, .12, R)
        self.torso.parent = self.body
        belly = uvs(f"{name}_belly", (0, -1.86, 4.5), (2.0, 1.1, 2.3), mat(f"{name}belly", "#5b7382", .55), self.body)
        # Head casting.
        self.head = empty(f"{name}_head", (0, 0, 8.5), self.body)
        hp = [uvs("h", (0, 0, 3.4), (3.75, 3.4, 3.45), seg=32, ring=20),
              uvs("cheekL", (-1.75, -1.75, 2.45), (1.75, 1.5, 1.45), seg=20, ring=12),
              uvs("cheekR", (1.75, -1.75, 2.45), (1.75, 1.5, 1.45), seg=20, ring=12),
              uvs("mz", (0, -2.75, 2.2), (2.5, 1.65, 1.5), seg=24, ring=16),
              uvs("earL", (-4.0, .2, 3.9), (1.5, .8, 1.65), seg=20, ring=12),
              uvs("earR", (4.0, .2, 3.9), (1.5, .8, 1.65), seg=20, ring=12)]
        self.skull = fuse(f"{name}_skull", hp, .09, R)
        self.skull.parent = self.head
        face = mat(f"{name}face", "#62798a", .5)
        self.parent_all([uvs(f"{name}_mzface", (0, -3.28, 2.15), (2.15, 1.25, 1.15), face),
                         uvs(f"{name}_faceL", (-1.28, -2.62, 3.92), (1.32, .92, 1.34), face),
                         uvs(f"{name}_faceR", (1.28, -2.62, 3.92), (1.32, .92, 1.34), face),
                         uvs(f"{name}_earInL", (-4.1, -.42, 3.9), (.92, .42, 1.05), face),
                         uvs(f"{name}_earInR", (4.1, -.42, 3.9), (.92, .42, 1.05), face),
                         uvs(f"{name}_nL", (-.34, -4.32, 2.65), (.13, .1, .13), pal["ink"]),
                         uvs(f"{name}_nR", (.34, -4.32, 2.65), (.13, .1, .13), pal["ink"])], self.head)
        self.eyes = [uvs(f"{name}_eyeL", (-1.28, -3.28, 3.9), (.72, .55, .78), pal["eye"], self.head),
                     uvs(f"{name}_eyeR", (1.28, -3.28, 3.9), (.72, .55, .78), pal["eye"], self.head)]
        self.brows = [tube(f"{name}_browL", [(-.8, 0, 0), (0, -.06, .14), (.8, 0, 0)], .2, R, self.head),
                      tube(f"{name}_browR", [(-.8, 0, 0), (0, -.06, .14), (.8, 0, 0)], .2, R, self.head)]
        self.brows[0].location = (-1.35, -3.46, 5.0)
        self.brows[1].location = (1.35, -3.46, 5.0)
        self.mouth = tube(f"{name}_mouth", [(-1.15, -4.3, 1.95), (0, -4.52, 1.55), (1.15, -4.3, 1.95)], .085, pal["ink"], self.head)
        self.maw = uvs(f"{name}_maw", (0, -4.28, 1.75), (.9, .35, .05), pal["ink"], self.head)
        if glasses:
            for sx in (-1, 1):
                bpy.ops.mesh.primitive_torus_add(major_radius=1.28, minor_radius=.12, major_segments=48, minor_segments=10,
                                                 location=(sx * 1.32, -3.78, 3.96), rotation=(math.pi / 2, 0, 0))
                g = bpy.context.object
                g.name = f"{name}_lens{sx}"
                bpy.ops.object.shade_smooth()
                g.data.materials.append(pal["green"])
                g.parent = self.head
            tube(f"{name}_bridge", [(-.2, -3.8, 4.05), (0, -3.98, 4.25), (.2, -3.8, 4.05)], .1, pal["green"], self.head)
        self.cap = None
        if nightcap:
            self.cap = self._nightcap()
        if visor:
            v = uvs(f"{name}_visor", (0, -3.0, 5.65), (3.2, 1.9, .24), pal["green"], self.head)
            cyl(f"{name}_band", 3.42, .62, (0, 0, 5.5), pal["dark"], self.head, bevel=.06)
        # Arms: shoulders are fixed in body space, hands are keyed.
        self.shL, self.shR = Vector((-1.5, -.1, 7.3)), Vector((1.5, -.1, 7.3))
        self.armL = tube(f"{name}_armL", [self.shL, self.shL + Vector((-1, 0, -2)), self.shL + Vector((-2, 0, -2)), Vector(self.state["L"])], .78, R, self.body)
        self.armR = tube(f"{name}_armR", [self.shR, self.shR + Vector((1, 0, -2)), self.shR + Vector((2, 0, -2)), Vector(self.state["R"])], .78, R, self.body)
        self.handL = self._hand("L")
        self.handR = self._hand("R")
        self.tail = tube(f"{name}_tail", [(0, 2.2, 3.2), (2.6, 3.6, 2.4), (5.2, 4.2, 4.2), (6.2, 3.8, 7.6), (4.6, 3.4, 9.2), (3.2, 3.6, 8.0)], .5, R, self.body, order=4)
        self._apply()

    def parent_all(self, objs, p):
        for o in objs:
            o.parent = p

    def _nightcap(self):
        c = uvs(f"{self.name}_capb", (0, .35, 6.2), (3.15, 3.0, 1.55), self.pal["steel"], self.head)
        t = tube(f"{self.name}_capt", [(0, .4, 7.2), (.4, .8, 8.6), (2.1, 1.1, 9.4), (4.4, 1.1, 8.8)], .95, self.pal["steel"], self.head)
        s = uvs(f"{self.name}_capp", (4.8, 1.1, 8.3), .68, self.pal["green"], self.head)
        return [c, t, s]

    def _hand(self, side):
        e = empty(f"{self.name}_hand{side}", self.state[side], self.body)
        uvs(f"{self.name}_palm{side}", (0, 0, 0), (.98, .9, .98), resin(), e)
        sx = 1 if side == "L" else -1
        uvs(f"{self.name}_thumb{side}", (sx * .65, -.55, .25), (.38, .38, .55), resin(), e)
        return e

    def _elbow(self, sh, hand, sign):
        sh, hand = Vector(sh), Vector(hand)
        mid = (sh + hand) * .5
        d = hand - sh
        # Rubber-hose bend: elbow drops down and out.
        out = Vector((sign * (1.0 + .18 * d.length), .5, -.24 * d.length))
        return sh + d * .28 + out * .9, sh + d * .68 + out * .62

    def _apply(self):
        st = self.state
        self.root.location = (st["pos"][0], st["pos"][1], st["hop"])
        self.root.rotation_euler = (0, math.radians(st["tip"]), math.radians(st["turn"]))
        self.body.rotation_euler = (math.radians(st["lean"][0]), math.radians(st["lean"][1]), 0)
        self.body.scale = (1 / math.sqrt(st["squash"]), 1 / math.sqrt(st["squash"]), st["squash"])
        y, p, r = st["head"]
        self.head.rotation_euler = (math.radians(p), math.radians(r), math.radians(y))
        for e in self.eyes:
            k = 1 - st["blink"]
            e.scale = (.72, .55, max(.05, .78 * k))
        for b, s in zip(self.brows, (-1, 1)):
            b.location.z = 5.0 + .34 * st["brow"]
            b.rotation_euler.y = math.radians(s * 14 * st["brow"])
        for side, sh, sign, arm, hand in (("L", self.shL, -1, self.armL, self.handL), ("R", self.shR, 1, self.armR, self.handR)):
            h = Vector(st[side])
            e1, e2 = self._elbow(sh, h, sign)
            pts = arm.data.splines[0].points
            pts[1].co, pts[2].co, pts[3].co = (*e1, 1), (*e2, 1), (*h, 1)
            hand.location = h
        pts = self.mouth.data.splines[0].points
        pts[1].co = (0, -4.52, 1.75 - .42 * st["smile"], 1)
        pts[0].co = (-1.15, -4.3, 1.75 + .2 * st["smile"], 1)
        pts[2].co = (1.15, -4.3, 1.75 + .2 * st["smile"], 1)
        self.maw.scale = (max(.05, .9 * st["open"]), .35, max(.03, .8 * st["open"]))

    def key(self, frame, **kw):
        """Set any pose values, then key the whole pose at this frame so holds are exact."""
        self.state.update(kw)
        self._apply()
        for o, attrs in ((self.root, ("location", "rotation_euler")), (self.body, ("rotation_euler", "scale")),
                         (self.head, ("rotation_euler",)), (self.handL, ("location",)), (self.handR, ("location",)),
                         (self.maw, ("scale",))):
            for a in attrs:
                o.keyframe_insert(a, frame=frame)
        for e in self.eyes:
            e.keyframe_insert("scale", frame=frame)
        for b in self.brows:
            b.keyframe_insert("location", frame=frame)
            b.keyframe_insert("rotation_euler", frame=frame)
        for arm in (self.armL, self.armR):
            for p in arm.data.splines[0].points:
                p.keyframe_insert("co", frame=frame)
        for p in self.mouth.data.splines[0].points:
            p.keyframe_insert("co", frame=frame)
        return self

    def hop_to(self, frame, pos, dur=10, height=3.5, **kw):
        """A miniature's hop: anticipation squash, arc, landing squash."""
        st0 = self.state["pos"]
        self.key(frame, squash=.86)
        self.key(frame + int(dur * .5), pos=((st0[0] + pos[0]) / 2, (st0[1] + pos[1]) / 2), hop=height, squash=1.08, **kw)
        self.key(frame + dur, pos=pos, hop=0, squash=.82)
        self.key(frame + dur + 4, squash=1.04)
        self.key(frame + dur + 7, squash=1.)
        return frame + dur + 7

    def hold(self, frame):
        return self.key(frame)

    def finish(self, amount=1.):
        """Hand-placed stop-motion jitter on the head and hands."""
        for o, path, st in ((self.head, "rotation_euler", .012), (self.handL, "location", .035), (self.handR, "location", .035)):
            if o.animation_data and o.animation_data.action:
                for fc in o.animation_data.action.fcurves:
                    if fc.data_path == path:
                        n = fc.modifiers.new("NOISE")
                        n.scale = 14
                        n.strength = st * amount
                        n.phase = hash(o.name + str(fc.array_index)) % 100

    def blink_at(self, frame):
        b = self.state["blink"]
        self.key(frame, blink=b)
        self.key(frame + 2, blink=.95)
        self.key(frame + 5, blink=b)

    def idle(self, a, b, every=48, seed=1):
        """Breathing and blinks over [a, b]; call last for the range."""
        import random
        rnd = random.Random(seed)
        f = a + rnd.randint(6, 30)
        while f < b - 6:
            self.blink_at(f)
            f += every + rnd.randint(-14, 24)

    def world(self, local):
        return self.root.location + Vector(local)
