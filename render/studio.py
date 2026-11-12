"""Renders the watch-only fencer from render/prepare.py --watch-only in a studio: a piste, a pool of light,
and a camera that orbits the fencer once over the frames.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P render/studio.py -- <take dir> <width> <height> [last frame [first frame]]
writes <take dir>/frames/f_NNNN.png.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
take, width, height = Path(args[0]).resolve(), int(args[1]), int(args[2])
s = json.loads((take / "scene.json").read_text())
first, n = s["first"], s["frames"]
start = int(args[4]) if len(args) > 4 else first
end = int(args[3]) if len(args) > 3 else first + n - 1

MINT = (0.18, 0.86, 0.60)
RADIUS, HEIGHT, LOOK_AT = 4.6, 1.25, (0.0, 0.0, 0.8)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = round(s["fps"])
scene.frame_start, scene.frame_end = start, end
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(take / "frames" / "f_")
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 24
scene.eevee.use_raytracing = True
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.35
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"

world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.004, 0.006, 1)
scene.world = world


def principled(name, rgb, rough, metallic=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def trail_material():
    """Glows at the tip and fades to nothing along the path: `age` is 0 at the tip, 1 at the tail."""
    m = bpy.data.materials.new("trail")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    age = nt.nodes.new("ShaderNodeAttribute")
    age.attribute_name = "age"
    fade = nt.nodes.new("ShaderNodeMath")
    fade.operation = "POWER"
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(age.outputs["Fac"], inv.inputs[1])
    nt.links.new(inv.outputs[0], fade.inputs[0])
    fade.inputs[1].default_value = 1.6
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*MINT, 1)
    em.inputs["Strength"].default_value = 14.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(fade.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def kit_material():
    """The fencing kit: colour, roughness and metal per vertex (render/prepare.py's kit), with the mask's wire
    mesh drawn where `kit_mesh` is positive so its edge is a clean line, not a row of triangles."""
    m = bpy.data.materials.new("kit")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    col, fin, mesh = (nt.nodes.new("ShaderNodeAttribute") for _ in range(3))
    col.attribute_name, fin.attribute_name, mesh.attribute_name = "kit_col", "kit_finish", "kit_mesh"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(fin.outputs["Color"], sep.inputs["Color"])
    edge = nt.nodes.new("ShaderNodeMapRange")
    edge.inputs["From Min"].default_value, edge.inputs["From Max"].default_value = -0.002, 0.002
    nt.links.new(mesh.outputs["Fac"], edge.inputs["Value"])
    wire = s["kit"]["mesh"]
    for socket, src, value in (("Base Color", col.outputs["Color"], (*wire[:3], 1)),
                               ("Roughness", sep.outputs["Red"], wire[3]), ("Metallic", sep.outputs["Green"], wire[4])):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA" if socket == "Base Color" else "FLOAT"
        a, bb, out = (6, 7, 2) if socket == "Base Color" else (2, 3, 0)
        nt.links.new(edge.outputs["Result"], mix.inputs[0])
        nt.links.new(src, mix.inputs[a])
        mix.inputs[bb].default_value = value
        nt.links.new(mix.outputs[out], b.inputs[socket])
    nt.links.new(sep.outputs["Blue"], b.inputs["Sheen Weight"])
    b.inputs["Sheen Roughness"].default_value = 0.4
    return m


def dress(ob):
    """Paints the kit and lets the cloth hang off the body: smoothed where it is fabric, then let out a little."""
    kit, part = s["kit"], s["body_kit"]
    me = ob.data
    col = me.attributes.new("kit_col", "FLOAT_COLOR", "POINT")
    col.data.foreach_set("color", [c for p in part for c in (*kit[p][:3], 1.0)])
    fin = me.attributes.new("kit_finish", "FLOAT_COLOR", "POINT")
    fin.data.foreach_set("color", [c for p in part for c in (kit[p][3], kit[p][4], 0.5 * kit[p][5], 1.0)])
    me.attributes.new("kit_mesh", "FLOAT", "POINT").data.foreach_set("value", s["body_mesh"])
    cloth = ob.vertex_groups.new(name="cloth")
    for i, p in enumerate(part):
        if kit[p][5] > 0:
            cloth.add([i], kit[p][5], "REPLACE")
    sm = ob.modifiers.new("smooth", "SMOOTH")
    sm.factor, sm.iterations, sm.vertex_group = 0.5, 30, "cloth"
    out = ob.modifiers.new("ease", "DISPLACE")
    out.strength, out.mid_level, out.vertex_group = 0.016, 0.0, "cloth"


def cached_mesh(name, rest, faces, pc2, mat, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(rest, [], faces)
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    mod = ob.modifiers.new("cache", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = str(take / pc2)
    mod.frame_start = first
    return ob


def plane(name, sx, sy, z, mat, yaw=0.0, x=0.0, y=0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    ob.location = (x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw), z)
    ob.rotation_euler = (0, 0, yaw)
    ob.data.materials.append(mat)
    scene.collection.objects.link(ob)
    return ob


lit = bpy.data.collections.new("fencer")
scene.collection.children.link(lit)
body = cached_mesh("body", s["body_rest"], s["body_faces"], "body.pc2", kit_material())
dress(body)
for ob in (body, cached_mesh("blade", s["blade_rest"], s["blade_faces"], "blade.pc2",
                             principled("blade", (0.9, 0.9, 0.92), 0.25, 1.0, (1, 1, 1), 0.8))):
    lit.objects.link(ob)
trail = cached_mesh("trail", s["trail_rest"], s["trail_faces"], "trail.pc2", trail_material())
trail.visible_shadow = False
trail.data.attributes.new("age", "FLOAT", "POINT").data.foreach_set("value", s["trail_age"])

# The piste runs along the line of the feet: 1.5 m wide, with its centre and en-garde lines.
yaw = math.radians(s["facing_deg"])
plane("floor", 400, 400, 0.0, principled("floor", (0.012, 0.012, 0.015), 0.42))
plane("piste", 7.0, 0.75, 0.002, principled("piste", (0.035, 0.036, 0.042), 0.45), yaw)
line = principled("line", (0.3, 0.3, 0.32), 0.5, 0.0, (0.75, 0.8, 0.8), 0.15)
for side in (-0.75, 0.75):
    plane("edge", 7.0, 0.012, 0.004, line, yaw, 0.0, side)
for x in (-2.0, 0.0, 2.0):
    plane("mark", 0.012, 0.75, 0.004, line, yaw, x, 0.0)


def light(name, kind, loc, energy, rgb, size, spot=None, parent=None):
    d = bpy.data.lights.new(name, kind)
    d.energy, d.color = energy, rgb
    if kind == "AREA":
        d.size = size
        d.specular_factor = 0.25  # or the glossy floor mirrors each panel as a white slab
    else:
        d.shadow_soft_size = size
        d.spot_size, d.spot_blend = spot, 0.9
    ob = bpy.data.objects.new(name, d)
    ob.location = loc
    scene.collection.objects.link(ob)
    c = ob.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    if parent:
        ob.parent = parent
        ob.light_linking.receiver_collection = lit  # the panels shape the fencer; only the pool lights the floor
    return ob


target = bpy.data.objects.new("target", None)
target.location = LOOK_AT
scene.collection.objects.link(target)
# The panels turn with the camera (which starts on the rig's +x): key beside it, mint rim always behind the fencer.
rig = bpy.data.objects.new("rig", None)
scene.collection.objects.link(rig)
light("pool", "SPOT", (0, 0, 7.0), 3600, (1.0, 0.96, 0.9), 0.6, math.radians(62))
light("key", "AREA", (3.5, -2.5, 3.2), 300, (1.0, 0.95, 0.88), 2.5, parent=rig)
light("rim", "AREA", (-3.5, 2.5, 2.4), 500, MINT, 2.0, parent=rig)
light("fill", "AREA", (3.0, 3.5, 1.2), 40, (0.6, 0.7, 1.0), 4.0, parent=rig)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 36.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
c = cam.constraints.new("TRACK_TO")
c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
# One full turn, starting side-on with the fencer facing left as the video did; slower into and out of the cut.
for f in range(first, first + n):
    u = (f - first) / (n - 1)
    a = yaw + math.pi / 2 + 2 * math.pi * (u - 0.5 * math.sin(2 * math.pi * u) / (2 * math.pi))
    cam.location = Vector((RADIUS * math.cos(a), RADIUS * math.sin(a), HEIGHT + 0.25 * math.sin(2 * math.pi * u)))
    cam.keyframe_insert("location", frame=f)
    rig.rotation_euler = (0, 0, a)
    rig.keyframe_insert("rotation_euler", frame=f)

# Bloom on the trail and the blade's highlights.
tree = bpy.data.node_groups.new("comp", "CompositorNodeTree")
scene.compositing_node_group = tree
tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl, glare, out = (tree.nodes.new(t) for t in ("CompositorNodeRLayers", "CompositorNodeGlare", "NodeGroupOutput"))
glare.inputs["Type"].default_value = "Bloom"
glare.inputs["Threshold"].default_value = 2.5
glare.inputs["Strength"].default_value = 0.35
glare.inputs["Size"].default_value = 0.5
tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
tree.links.new(glare.outputs["Image"], out.inputs["Image"])

bpy.ops.render.render(animation=True)
