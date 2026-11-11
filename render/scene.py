"""Builds and renders the overlay scene headless.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P render/scene.py -- <take dir> <width> <height> [last frame]
renders RGBA frames to <take dir>/frames/, then composite.sh lays them over the video.
    ... -- <take dir> <width> <height> --blend <video>
instead saves <take dir>/scene.blend with the video as the camera background, for viewing in Blender.
"""

import json
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
video = None
if "--blend" in args:
    video = Path(args[args.index("--blend") + 1]).resolve()
    args = args[:args.index("--blend")]
take, width, height = Path(args[0]).resolve(), int(args[1]), int(args[2])
last = int(args[3]) if len(args) > 3 else None
s = json.loads((take / "scene.json").read_text())

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = round(s["fps"])
scene.frame_start, scene.frame_end = 1, last or s["frames"]
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = str(take / "frames" / "f_")
scene.render.engine = "BLENDER_EEVEE"

cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = s["fx"] * 36.0 / width
cam_data.shift_x = -(s["cx"] - width / 2) / width
cam_data.shift_y = (s["cy"] - height / 2) / width
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def material(name, rgb, alpha, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1)
    em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = alpha
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def cached_mesh(name, rest, faces, pc2, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(rest, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    mod = ob.modifiers.new("cache", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = str(take / pc2)
    mod.frame_start = 1
    return ob


body = cached_mesh("body", s["body_rest"], s["body_faces"], "body.pc2",
                   material("body", (0.55, 0.75, 1.0), 0.45, 1.2))
for p in body.data.polygons:
    p.use_smooth = True

blade_faces = [(0, 1, 3, 2), (4, 5, 7, 6)]
cached_mesh("blade", s["blade_rest"], blade_faces, "blade.pc2", material("blade", (0.9, 0.9, 0.95), 1.0, 2.0))

n = s["trail_frames"]
trail_faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
cached_mesh("trail", s["trail_rest"], trail_faces, "trail.pc2", material("trail", (1.0, 0.35, 0.1), 0.9, 6.0))

if "skeleton_rest" in s:
    cached_mesh("skeleton", s["skeleton_rest"], s["skeleton_faces"], "skeleton.pc2",
                material("skeleton", (0.1, 1.0, 0.3), 0.95, 1.0))

if video:
    clip = bpy.data.movieclips.load(str(video))
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    bg.alpha = 1.0
    bpy.ops.wm.save_as_mainfile(filepath=str(take / "scene.blend"))
else:
    bpy.ops.render.render(animation=True)
