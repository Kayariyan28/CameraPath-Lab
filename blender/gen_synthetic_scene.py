"""Render synthetic reference videos with KNOWN camera trajectories.

Runs inside Blender's own interpreter:

    blender --background --python blender/gen_synthetic_scene.py -- \
        --scene dolly_forward --out benchmarks/synthetic --frames 90 --fps 30

This is the backbone of the validation suite (spec §24). Claims about a camera
solver are worth nothing without ground truth, and Blender can produce it: a
textured static scene, a camera on an exactly known path, and the true pose of
every frame written out beside the MP4.

The scene is built for *measurability*, not beauty:
  * high-frequency procedural texture everywhere, so features are trackable
  * geometry at many depths, so translation produces real parallax — without
    depth variation a dolly is indistinguishable from a zoom, and the test would
    be measuring nothing
  * no moving content except where a case deliberately adds it
  * flat, even lighting, so tracking is not defeated by its own shadows

Ground truth is written in *Blender's* convention (location + wxyz quaternion).
Converting it is the backend's job, using the tested functions in
geometry/conventions.py — so the comparison exercises that code rather than
duplicating it here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def noise_material(name: str, scale: float = 14.0, detail: float = 10.0,
                   base: float = 0.5) -> bpy.types.Material:
    """Matte material with a high-frequency procedural pattern.

    Two noise octaves at different scales are multiplied together, which gives
    corner-like structure at several spatial frequencies. A single smooth noise
    gives gradients that Lucas-Kanade cannot localise, and a purely random
    per-pixel pattern aliases badly under motion — this sits between them.

    Deliberately NOT self-similar in the fractal sense: scale-invariant texture
    is the adversarial worst case for both LK and SIFT matching.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    coords = nodes.new("ShaderNodeTexCoord")

    noise_a = nodes.new("ShaderNodeTexNoise")
    noise_a.inputs["Scale"].default_value = scale
    noise_a.inputs["Detail"].default_value = detail

    noise_b = nodes.new("ShaderNodeTexNoise")
    noise_b.inputs["Scale"].default_value = scale * 5.7  # non-integer ratio
    noise_b.inputs["Detail"].default_value = 4.0

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.65

    ramp = nodes.new("ShaderNodeValToRGB")
    # Steepen the contrast so features are crisp rather than washed out.
    ramp.color_ramp.elements[0].position = 0.32
    ramp.color_ramp.elements[1].position = 0.68
    ramp.color_ramp.elements[0].color = (base * 0.35, base * 0.35, base * 0.35, 1.0)
    ramp.color_ramp.elements[1].color = (base * 1.5, base * 1.5, base * 1.5, 1.0)

    links.new(coords.outputs["Object"], noise_a.inputs["Vector"])
    links.new(coords.outputs["Object"], noise_b.inputs["Vector"])
    links.new(noise_a.outputs["Fac"], mix.inputs["Color1"])
    links.new(noise_b.outputs["Fac"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def build_world(seed: int = 7) -> None:
    """A static, textured environment the camera travels THROUGH.

    Depth distribution is the whole point, and the first version of this scene
    got it wrong in a way worth recording. Geometry was placed from y=3 to y=70
    while the camera path ran from y=-26 to y=-4, so the nearest object was 29 m
    away and everything else further. The result: 22 m of genuine camera
    translation produced a median image flow of only 1.55 px/frame, the
    triangulation baseline was negligible, and COLMAP could not even find an
    initial image pair ("No good initial image pair found").

    Structure-from-motion needs near geometry. Parallax is the DIFFERENCE in
    apparent motion between near and far content, so a scene that is uniformly
    distant has almost none however far the camera moves. Objects are therefore
    now distributed from behind the camera's start to well beyond its end, so the
    camera flies through the field and near objects sweep past while distant ones
    barely move.

    A corridor around the camera paths is kept clear so the camera does not end
    up inside an object, which would occlude the frame and defeat the test.
    """
    import random
    rng = random.Random(seed)

    ground_mat = noise_material("ground", scale=7.0, detail=12.0, base=0.62)
    block_mat = noise_material("blocks", scale=16.0, detail=9.0, base=0.72)
    far_mat = noise_material("far", scale=3.0, detail=10.0, base=0.5)

    # Ground.
    bpy.ops.mesh.primitive_plane_add(size=300, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(ground_mat)

    # Distant walls: structure in the far field, so a purely rotational move
    # still has something to track.
    for i, (loc, rot, size) in enumerate([
        ((0, 110, 26), (math.pi / 2, 0, 0), 220),
        ((0, -110, 26), (math.pi / 2, 0, 0), 220),
        ((110, 0, 26), (math.pi / 2, 0, math.pi / 2), 220),
        ((-110, 0, 26), (math.pi / 2, 0, math.pi / 2), 220),
    ]):
        bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot)
        wall = bpy.context.active_object
        wall.name = f"wall_{i}"
        wall.data.materials.append(far_mat)

    def clear_of_camera(x: float, y: float, radius: float) -> bool:
        """Keep a corridor free along the camera paths used by the scenes.

        Paths run roughly along x=0 for y in [-30, 2] (dollies, pans, tilts),
        across x in [-15, 15] at y=-18 (trucking), and around a 20 m circle about
        (0, 14) (orbit). An object inside the camera would black out the frame.
        """
        if abs(x) < 3.0 + radius and -32.0 < y < 3.0:
            return False
        if abs(y + 18.0) < 3.0 + radius and abs(x) < 17.0:
            return False
        orbit_r = math.hypot(x, y - 14.0)
        if abs(orbit_r - 20.0) < 2.5 + radius:
            return False
        return True

    # Mid-field blocks spanning the camera's own depth range and beyond. This is
    # where parallax comes from: near blocks sweep past, far blocks crawl.
    placed = 0
    attempts = 0
    while placed < 190 and attempts < 3000:
        attempts += 1
        depth = rng.uniform(-34.0, 85.0)
        lateral = rng.uniform(-50.0, 50.0)
        # Bias towards the near field, where parallax is strongest, without
        # leaving the far field empty.
        if rng.random() < 0.45:
            depth = rng.uniform(-30.0, 12.0)
            lateral = rng.uniform(-16.0, 16.0)
        sx, sy = rng.uniform(0.5, 3.2), rng.uniform(0.5, 3.2)
        height = rng.uniform(0.7, 8.0)
        if not clear_of_camera(lateral, depth, max(sx, sy) * 0.75):
            continue
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=(lateral, depth, height / 2.0),
            rotation=(0, 0, rng.uniform(0, math.pi)),
        )
        block = bpy.context.active_object
        block.name = f"block_{placed}"
        block.scale = (sx, sy, height)
        block.data.materials.append(block_mat)
        placed += 1

    # Vertical poles: strongly localised features at known depths, including
    # very close ones.
    placed = 0
    attempts = 0
    while placed < 60 and attempts < 1500:
        attempts += 1
        depth = rng.uniform(-32.0, 70.0)
        lateral = rng.uniform(-40.0, 40.0)
        radius = rng.uniform(0.10, 0.30)
        if not clear_of_camera(lateral, depth, radius + 0.6):
            continue
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=rng.uniform(4.0, 15.0),
            location=(lateral, depth, 5.0), vertices=14,
        )
        pole = bpy.context.active_object
        pole.name = f"pole_{placed}"
        pole.data.materials.append(block_mat)
        placed += 1

    # Spheres: features that look the same from every angle, which is a
    # different matching problem from a cube's corners.
    placed = 0
    attempts = 0
    while placed < 44 and attempts < 1200:
        attempts += 1
        depth = rng.uniform(-30.0, 65.0)
        lateral = rng.uniform(-36.0, 36.0)
        radius = rng.uniform(0.4, 1.8)
        if not clear_of_camera(lateral, depth, radius + 0.5):
            continue
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=radius, subdivisions=2,
            location=(lateral, depth, rng.uniform(0.6, 7.0)),
        )
        sphere = bpy.context.active_object
        sphere.name = f"sphere_{placed}"
        sphere.data.materials.append(block_mat)
        placed += 1

    # Lighting: bright and directional enough to give the procedural texture
    # real local contrast. The first version rendered washed-out and hazy, which
    # depresses the structure tensor Lucas-Kanade depends on.
    bpy.ops.object.light_add(type="SUN", location=(30, -40, 60))
    sun = bpy.context.active_object
    sun.data.energy = 5.0
    sun.data.angle = 0.25  # tighter = crisper shading detail
    sun.rotation_euler = Euler((math.radians(52), math.radians(16), math.radians(-42)))

    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        # Dimmer ambient than the first version: strong uniform ambient flattens
        # everything towards mid-grey and destroys trackable contrast.
        bg.inputs[0].default_value = (0.32, 0.35, 0.40, 1.0)
        bg.inputs[1].default_value = 0.55


def add_moving_object(frames: int, fps: int) -> None:
    """One large object crossing the frame, for the dynamic-rejection case."""
    mat = noise_material("mover", scale=30.0, detail=6.0, base=0.8)
    bpy.ops.mesh.primitive_cube_add(size=3.2, location=(-14, 9, 1.8))
    mover = bpy.context.active_object
    mover.name = "MOVING_OBJECT"
    mover.data.materials.append(mat)
    mover.rotation_mode = "QUATERNION"
    for f in range(frames):
        t = f / max(frames - 1, 1)
        mover.location = Vector((-14 + 28 * t, 9 + 2.0 * math.sin(t * 3.0), 1.8))
        mover.keyframe_insert("location", frame=f + 1)



def iter_fcurves(action):
    """Yield every F-curve in an action, across Blender API generations.

    Blender 4.4 replaced the flat `action.fcurves` list with layered, slotted
    actions (`action.layers[].strips[].channelbags[].fcurves`). Both spellings
    are handled so the generator runs on current and older Blender alike.
    """
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                for bag in getattr(strip, "channelbags", []) or []:
                    yield from getattr(bag, "fcurves", []) or []
        return
    yield from getattr(action, "fcurves", []) or []


# ---------------------------------------------------------------------------
# Camera paths. Each returns (location, look_at_target, roll_radians, lens_mm).
# ---------------------------------------------------------------------------


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def camera_path(scene: str, t: float, lens_base: float = 28.0):
    """Camera state at normalised time t in [0, 1]."""
    if scene == "dolly_forward":
        return Vector((0, -26 + 22 * t, 2.4)), Vector((0, 30, 2.4)), 0.0, lens_base
    if scene == "dolly_backward":
        return Vector((0, -4 - 22 * t, 2.4)), Vector((0, 30, 2.4)), 0.0, lens_base
    if scene == "truck_right":
        return Vector((-14 + 28 * t, -18, 2.4)), Vector((-14 + 28 * t, 30, 2.4)), 0.0, lens_base
    if scene == "pedestal_up":
        return Vector((0, -18, 1.0 + 9.0 * t)), Vector((0, 30, 1.0 + 9.0 * t)), 0.0, lens_base
    if scene == "pan":
        a = math.radians(-34 + 68 * t)
        eye = Vector((0, -16, 2.6))
        return eye, eye + Vector((math.sin(a) * 40, math.cos(a) * 40, 0)), 0.0, lens_base
    if scene == "tilt":
        a = math.radians(-16 + 34 * t)
        eye = Vector((0, -16, 3.0))
        return eye, eye + Vector((0, 40, math.tan(a) * 40)), 0.0, lens_base
    if scene == "roll":
        return Vector((0, -16, 2.6)), Vector((0, 30, 2.6)), math.radians(48 * t), lens_base
    if scene == "orbit":
        a = math.radians(-52 + 104 * t)
        r = 20.0
        return (
            Vector((math.sin(a) * r, 14 - math.cos(a) * r, 4.2)),
            Vector((0, 14, 2.0)), 0.0, lens_base,
        )
    if scene == "crane_up":
        return (
            Vector((0, -20 + 6 * t, 1.4 + 13 * t)),
            Vector((0, 26, 2.0)), 0.0, lens_base,
        )
    if scene == "rise_and_tilt":
        z = 1.4 + 13 * t
        return Vector((0, -19, z)), Vector((0, 26, 1.0)), 0.0, lens_base
    if scene == "diagonal_flythrough":
        return (
            Vector((-13 + 26 * t, -24 + 34 * t, 3.0 + 3.0 * t)),
            Vector((6, 34, 2.0)), 0.0, lens_base,
        )
    if scene == "fpv_curve":
        a = t * math.pi * 0.85
        eye = Vector((math.sin(a) * 15 - 6, -22 + 30 * t, 2.2 + 2.6 * math.sin(a * 1.4)))
        ahead = t + 0.06
        a2 = min(ahead, 1.0) * math.pi * 0.85
        target = Vector((
            math.sin(a2) * 15 - 6,
            -22 + 30 * min(ahead, 1.0) + 9.0,
            2.2 + 2.6 * math.sin(a2 * 1.4),
        ))
        return eye, target, math.radians(13 * math.sin(a * 1.8)), lens_base
    if scene == "accelerating":
        return Vector((0, -26 + 22 * _ease_in(t), 2.4)), Vector((0, 30, 2.4)), 0.0, lens_base
    if scene == "decelerating":
        return Vector((0, -26 + 22 * _ease_out(t), 2.4)), Vector((0, 30, 2.4)), 0.0, lens_base
    if scene == "handheld":
        # Broadband, non-periodic wobble riding on a slow push in.
        j = (
            0.10 * math.sin(t * 41.0) + 0.06 * math.sin(t * 67.0 + 1.1)
            + 0.04 * math.sin(t * 113.0 + 0.4)
        )
        k = (
            0.09 * math.cos(t * 37.0) + 0.05 * math.sin(t * 71.0 + 2.0)
            + 0.03 * math.cos(t * 97.0)
        )
        eye = Vector((j * 2.2, -20 + 9 * t, 2.5 + k * 1.6))
        return eye, Vector((j * 5.0, 30, 2.4 + k * 3.0)), math.radians(2.2 * j * 9), lens_base
    if scene == "zoom_only":
        # Camera fixed; only the lens changes. The degenerate case (spec §25).
        return Vector((0, -20, 2.5)), Vector((0, 30, 2.5)), 0.0, lens_base * (1.0 + 1.25 * t)
    if scene == "dolly_zoom":
        # Move in while zooming out to hold subject size — the classic case
        # where translation and focal change must be separated.
        return (
            Vector((0, -26 + 14 * t, 2.4)), Vector((0, 30, 2.4)), 0.0,
            lens_base * (1.0 - 0.42 * t),
        )
    if scene == "static":
        return Vector((0, -18, 2.5)), Vector((0, 30, 2.5)), 0.0, lens_base
    if scene == "moving_object":
        return Vector((0, -22 + 10 * t, 2.4)), Vector((0, 26, 2.2)), 0.0, lens_base
    raise SystemExit(f"unknown scene '{scene}'")


SCENES = [
    "dolly_forward", "dolly_backward", "truck_right", "pedestal_up",
    "pan", "tilt", "roll", "orbit", "crane_up", "rise_and_tilt",
    "diagonal_flythrough", "fpv_curve", "accelerating", "decelerating",
    "handheld", "zoom_only", "dolly_zoom", "static", "moving_object",
]


def setup_camera(scene_name: str, frames: int, fps: int, lens_base: float):
    """Create and animate the camera, returning the ground-truth records."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"

    truth = []
    for f in range(frames):
        t = f / max(frames - 1, 1)
        eye, target, roll, lens = camera_path(scene_name, t, lens_base)

        direction = (target - eye)
        if direction.length < 1e-9:
            direction = Vector((0, 1, 0))
        # Blender cameras look down local -Z, so the rotation is built from the
        # negated view direction.
        quat = (-direction).normalized().to_track_quat("Z", "Y")
        if abs(roll) > 1e-12:
            from mathutils import Quaternion
            quat = quat @ Quaternion(Vector((0, 0, 1)), roll)

        cam.location = eye
        cam.rotation_quaternion = quat
        cam_data.lens = lens

        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_quaternion", frame=f + 1)
        cam_data.keyframe_insert("lens", frame=f + 1)

        fov_h = 2.0 * math.atan(cam_data.sensor_width / (2.0 * lens))
        truth.append({
            "frame": f,
            "time": f / fps,
            "location": [eye.x, eye.y, eye.z],
            "quaternion_wxyz": [quat.w, quat.x, quat.y, quat.z],
            "lens_mm": lens,
            "sensor_width_mm": cam_data.sensor_width,
            "fov_horizontal": math.degrees(fov_h),
        })

    # Linear interpolation on every keyframe: Blender's default Bezier easing
    # would silently ease every move in and out, so an "accelerating" case would
    # not actually be the acceleration written into the ground truth. This is
    # load-bearing for the timing and speed tests, not a cosmetic setting.
    for action_owner in (cam, cam_data):
        anim = action_owner.animation_data
        if not anim or not anim.action:
            continue
        for fcurve in iter_fcurves(anim.action):
            for kp in fcurve.keyframe_points:
                kp.interpolation = "LINEAR"

    return cam, truth


def configure_render(width: int, height: int, frames: int, fps: int,
                     out_path: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = pick_render_engine()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = frames

    # Blender 5 gates the video formats behind media_type; on that build
    # file_format="FFMPEG" is rejected until media_type is VIDEO.
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    # High quality on purpose: compression artefacts would be measured as
    # tracking error and contaminate the accuracy numbers.
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = 12
    scene.render.filepath = out_path
    scene.render.use_overwrite = True

    if scene.render.engine.startswith("BLENDER_EEVEE"):
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass
        # Motion blur off: it would make the ground truth untrackable and is not
        # what we are testing.
        try:
            scene.render.use_motion_blur = False
        except AttributeError:
            pass


def available_render_engines() -> list[str]:
    """Engine identifiers this Blender build actually offers.

    Read from the RNA enum rather than hard-coded: the identifier has changed
    across releases (EEVEE -> BLENDER_EEVEE_NEXT -> BLENDER_EEVEE again), and
    assigning a name that does not exist raises at render time.
    """
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        return [item.identifier for item in prop.enum_items]
    except Exception:  # noqa: BLE001
        return ["BLENDER_WORKBENCH"]


def pick_render_engine() -> str:
    """Fastest engine that still shades the procedural textures.

    EEVEE is preferred: it is raster-fast and honours the noise materials that
    make the scene trackable. Workbench is the fallback — it renders flat matcap
    shading, which produces far fewer trackable features, so it is a last resort
    rather than an equivalent choice. Cycles is avoided as far too slow for a
    19-scene benchmark.
    """
    engines = available_render_engines()
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in engines:
            return candidate
    return "BLENDER_WORKBENCH"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help=f"one of: {', '.join(SCENES)}, or 'all'")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lens", type=float, default=28.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    scenes = SCENES if args.scene == "all" else [args.scene]
    os.makedirs(args.out, exist_ok=True)

    for name in scenes:
        print(f"[gen] building {name}", flush=True)
        clear_scene()
        build_world(seed=args.seed)
        if name == "moving_object":
            add_moving_object(args.frames, args.fps)

        _, truth = setup_camera(name, args.frames, args.fps, args.lens)

        video_path = os.path.join(args.out, f"{name}.mp4")
        configure_render(
            args.width, args.height, args.frames, args.fps,
            os.path.join(args.out, name), args.samples,
        )
        bpy.ops.render.render(animation=True)

        # Blender appends a frame range to FFMPEG output names.
        produced = None
        for candidate in os.listdir(args.out):
            if candidate.startswith(name) and candidate.endswith(".mp4"):
                produced = os.path.join(args.out, candidate)
                break
        if produced and produced != video_path:
            os.replace(produced, video_path)

        meta = {
            "scene": name,
            "fps": args.fps,
            "frame_count": args.frames,
            "resolution": [args.width, args.height],
            "convention": {
                "note": (
                    "Ground truth is in BLENDER convention: right-handed Z-up world, "
                    "camera looks along local -Z with local +Y up. Quaternions are "
                    "[w,x,y,z] camera-to-world. Convert with "
                    "geometry.conventions.blender_quat_to_camerapath."
                ),
                "world_up": "+Z",
                "camera_forward": "-Z",
                "camera_up": "+Y",
                "quaternion_order": "wxyz",
            },
            "video": os.path.basename(video_path),
            "frames": truth,
        }
        with open(os.path.join(args.out, f"{name}.truth.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"[gen] wrote {video_path} + ground truth ({len(truth)} poses)", flush=True)


if __name__ == "__main__":
    main()
