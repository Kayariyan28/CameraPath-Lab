"""Film a neutral 3D motion cage with the recovered camera. THE deliverable.

Runs inside Blender's own interpreter:

    blender --background --python blender/render_motion_proxy.py -- \
        --trajectory workspace/jobs/<id>/outputs/trajectory.json \
        --out workspace/jobs/<id>/outputs/motion_proxy.mp4 \
        --style motion_cage --width 1920 --height 1080

The product is the *camera move*, so the scene exists only to make that move
legible (spec §16-18). It is built for one reader: a generative-video model that
is given this clip as a motion reference. That reader can only infer motion from
what changes between frames, which fixes the whole design:

  * **Depth layering is the entire point.** Translation is invisible without
    objects at several distances — a dolly past a flat backdrop is pixel-for-pixel
    a zoom. So every style keeps near, mid and far anchors, and the scattered
    geometry is placed *along the camera path* rather than around the origin, so
    the near field stays populated for the whole shot however far the camera
    travels.
  * **Repeated markers give absolute speed.** A regular ground grid turns
    translation into a countable rate; without it speed is only relative.
  * **A horizon and a ground plane give rotation an absolute frame**, so tilt and
    roll are unambiguous rather than being read as translation.
  * **Equal-angular-size depth ladders.** The ladder spheres grow with distance
    so they subtend the same angle on screen; parallax is then the only cue that
    separates them, which is exactly the cue we want exercised.

It deliberately does not resemble any real place: monochrome greys, primitive
solids, flat lighting, no signage, no sky texture. No source pixel reaches it —
the only thing carried over from the reference video is the camera's motion.

Coordinate handling: the trajectory JSON is in CameraPath convention (right-handed
Z-up world, camera looks along +Y with +Z up, quaternions [w,x,y,z] camera→world).
`camerapath_quat_to_blender()` below reimplements
`app.geometry.conventions.camerapath_quat_to_blender` because Blender's
interpreter cannot import the backend package. It is the same matrix identity,
and `tests/unit/test_blender_runner.py` cross-checks this implementation against
that tested function through `--dump-poses` (invariant I8).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STYLES = ("motion_cage", "depth_poles", "ground_grid", "minimal")

#: Sensor width the virtual camera is built on. Any value works as long as the
#: focal length is derived from it consistently; 36 mm makes `lens` readable as
#: a full-frame equivalent for anyone opening the exported .blend.
SENSOR_WIDTH_MM = 36.0

#: Horizontal FOV used when a frame carries none. Matches
#: `app.geometry.intrinsics.DEFAULT_HORIZONTAL_FOV`; kept in sync by hand
#: because the backend package is not importable here.
DEFAULT_HORIZONTAL_FOV = 60.0

#: Hard clamps, not plausibility bounds. `geometry/intrinsics.py` rejects
#: *estimates* outside 8-150 degrees; by the time a value reaches this script it
#: is a decision already made upstream, so we render it and only refuse the
#: mathematically degenerate edges where the focal length blows up.
MIN_RENDERABLE_FOV = 1.0
MAX_RENDERABLE_FOV = 179.0

#: The path span a normalized trajectory is scaled to (§7 targets 10-25 units).
#: Every cage dimension is expressed as a multiple of this, so the cage tracks
#: the trajectory's magnitude instead of assuming normalized units.
REFERENCE_SPAN = 16.0

#: Cage-unit multiplier bounds. Below the lower bound the geometry is too small
#: for the depth buffer to separate; above the upper one the scene stops fitting
#: in float precision.
MIN_CAGE_SCALE = 0.05
MAX_CAGE_SCALE = 10_000.0

#: How far below the lowest camera position the floor sits, in cage units. The
#: camera of a normalized trajectory starts at the origin, so a floor at z=0
#: would put the first frame exactly in the ground plane.
FLOOR_CLEARANCE = 2.5

#: Minimum horizontal gap between any scattered object and every sampled camera
#: position, in cage units before scaling. Objects were first placed only
#: relative to ONE random path sample, so on a curving move a block could land on
#: the route elsewhere and fill half the frame at the moment it passed — hiding
#: the very motion cues the proxy exists to show.
CAMERA_CLEARANCE = 2.5

#: Placement attempts before an object is skipped rather than forced into the path.
PLACEMENT_ATTEMPTS = 16

#: Largest half-extent an object may have, as a fraction of its distance to the
#: NEAREST camera position. Keeping objects off the path was not enough: a block
#: 7.1 units across standing 10.6 units away covered half the frame for a third of
#: the shot. With this cap an object's full width subtends at most
#: 2*atan(0.18) = 20 deg from anywhere on the path — under a third of a typical
#: 60-65 deg view — so near geometry keeps its parallax without hiding everything
#: behind it. Objects are only ever shrunk, never grown.
MAX_ANGULAR_HALF_EXTENT = 0.18

#: Backdrop cylinder radius in cage units. Large enough that it reads as "far
#: away" rather than as a room the camera is standing in.
BACKDROP_RADIUS = 200.0

#: Non-uniformity of source frame timing above which we say so. A CFR stream
#: shows well under 1%; 2% matches `video/ffprobe.VFR_JITTER_THRESHOLD`.
TIMING_JITTER_WARN = 0.02

#: Monochrome palette. Values are linear diffuse albedo; the depth anchors are
#: separated by lightness alone (near bright, far dark) so the render stays
#: monochrome while still coding depth.
GREYS = {
    "floor": 0.14,
    "grid_minor": 0.42,
    "grid_major": 0.72,
    "backdrop": 0.26,
    "horizon": 0.78,
    "pole": 0.50,
    "block": 0.58,
    "near": 0.82,
    "mid": 0.52,
    "far": 0.27,
}


class TrajectoryError(RuntimeError):
    """The trajectory file cannot be rendered. Always fatal, never patched over:
    substituting a plausible pose for a broken one would fabricate motion."""


# ---------------------------------------------------------------------------
# Trajectory input (spec §19 format)
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """One camera pose, still in CameraPath convention."""

    frame: int
    time: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # [w, x, y, z], camera->world
    fov: float  # horizontal, degrees


@dataclass
class Trajectory:
    poses: list[Pose]
    fps: float
    scale_mode: str = "normalized"
    coordinate_system: dict = field(default_factory=dict)
    video: dict = field(default_factory=dict)
    shot_count: int = 1

    @property
    def duration(self) -> float:
        return len(self.poses) / self.fps if self.fps > 0 else 0.0


def _as_floats(value, count: int, what: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise TrajectoryError(f"{what} must be a list of {count} numbers, got {value!r}")
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(f"{what} contains a non-numeric entry: {value!r}") from exc
    if any(math.isnan(v) or math.isinf(v) for v in out):
        raise TrajectoryError(f"{what} contains NaN or infinity: {value!r}")
    return out


def _pick(record: dict, *names, default=None):
    """First present key from `names`. The exporter's field names have aliases
    (`frame`/`frame_index`, `fov`/`fov_horizontal`) and reading a trajectory
    should not depend on which spelling a given writer used."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _parse_pose(record: dict, index: int) -> Pose:
    if not isinstance(record, dict):
        raise TrajectoryError(f"frame {index} is not an object: {record!r}")

    position = _as_floats(
        _pick(record, "position", "location"), 3, f"frame {index} position"
    )
    quat = _as_floats(
        _pick(record, "quaternion", "quaternion_wxyz", "rotation_quaternion"),
        4, f"frame {index} quaternion",
    )
    norm = math.sqrt(sum(c * c for c in quat))
    if norm < 1e-9:
        raise TrajectoryError(
            f"frame {index} has a zero-length quaternion; that is not an "
            "orientation and cannot be rendered"
        )
    quat = tuple(c / norm for c in quat)

    fov = _pick(record, "fov", "fov_horizontal", "fov_horizontal_degrees")
    try:
        fov = float(fov) if fov is not None else DEFAULT_HORIZONTAL_FOV
    except (TypeError, ValueError):
        raise TrajectoryError(f"frame {index} has a non-numeric fov: {fov!r}") from None
    if not math.isfinite(fov) or fov <= 0.0:
        fov = DEFAULT_HORIZONTAL_FOV

    frame_number = _pick(record, "frame", "frame_index", default=index)
    time_value = _pick(record, "time", "timestamp", "time_seconds")
    try:
        frame_number = int(frame_number)
    except (TypeError, ValueError):
        frame_number = index
    try:
        time_value = float(time_value) if time_value is not None else float("nan")
    except (TypeError, ValueError):
        time_value = float("nan")

    return Pose(frame_number, time_value, position, quat, fov)


def load_trajectory(path: str) -> Trajectory:
    """Read the spec §19 trajectory export.

    A multi-shot export (`shots: [{frames: [...]}]`) is concatenated in order.
    Each shot is an independent coordinate system (invariant I4) and the seam is
    therefore a hard jump in the rendered camera — which is the correct output,
    because the seam is a cut in the source. Concatenating is what keeps the
    proxy the same length as the source (I1); rendering one shot per file would
    not.
    """
    if not os.path.isfile(path):
        raise TrajectoryError(f"trajectory file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryError(f"cannot read trajectory {path}: {exc}") from exc

    if not isinstance(doc, dict):
        raise TrajectoryError("trajectory root must be a JSON object")

    raw_frames = doc.get("frames")
    shot_count = 1
    if raw_frames is None:
        shots = doc.get("shots") or []
        if not isinstance(shots, list):
            raise TrajectoryError("'shots' must be a list")
        raw_frames = []
        for shot in shots:
            if not isinstance(shot, dict):
                raise TrajectoryError(f"shot entry is not an object: {shot!r}")
            raw_frames.extend(shot.get("frames") or shot.get("poses") or [])
        shot_count = max(1, len(shots))
    if not isinstance(raw_frames, list) or not raw_frames:
        raise TrajectoryError(
            "trajectory contains no frames; nothing to render "
            "(expected a top-level 'frames' list, or 'shots[].frames')"
        )

    poses = [_parse_pose(record, i) for i, record in enumerate(raw_frames)]

    fps = doc.get("fps")
    try:
        fps = float(fps) if fps is not None else 0.0
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0.0:
        fps = _fps_from_times(poses)
    if fps <= 0.0:
        raise TrajectoryError(
            "trajectory declares no usable fps and its frame times do not imply "
            "one; output timing cannot be reconstructed (invariant I1)"
        )

    return Trajectory(
        poses=poses,
        fps=fps,
        scale_mode=str(doc.get("scale_mode") or "normalized"),
        coordinate_system=doc.get("coordinate_system") or {},
        video=doc.get("video") or {},
        shot_count=shot_count,
    )


def _fps_from_times(poses: list[Pose]) -> float:
    """Recover a frame rate from the frame timestamps, as a last resort."""
    times = [p.time for p in poses if math.isfinite(p.time)]
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    if span <= 0.0:
        return 0.0
    return (len(times) - 1) / span


def scene_fps_pair(fps: float) -> tuple[int, float]:
    """Blender's (fps, fps_base) pair reproducing `fps` exactly.

    Blender stores the rate as an integer numerator over a float base, so
    23.976023976... is 24 / 1.001 rather than a rounded 24. Getting this wrong
    is a silent I1 violation: a 24/1 timebase on 23.976 fps material shortens a
    10-second clip by 10 ms per second.
    """
    if fps <= 0.0:
        raise TrajectoryError(f"fps must be positive, got {fps}")
    nearest = max(1, int(round(fps)))
    if abs(fps - nearest) < 1e-9:
        return nearest, 1.0
    return nearest, nearest / fps


# ---------------------------------------------------------------------------
# Coordinate conversion (mirror of app.geometry.conventions)
# ---------------------------------------------------------------------------

#: Columns are the Blender camera's local axes written in CameraPath-local
#: coordinates: Blender is [right, up, backward] = [+x_cpl, +z_cpl, -y_cpl].
#: Identical to `app.geometry.conventions.BLENDER_AXES_IN_CPL`; mathutils takes
#: rows, so the literal below is that matrix written out row by row.
BLENDER_AXES_IN_CPL = Matrix((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))


def camerapath_quat_to_blender(quat_wxyz: tuple[float, float, float, float]) -> Quaternion:
    """CameraPath camera→world quaternion -> Blender `rotation_quaternion`.

    Same identity as `app.geometry.conventions.camerapath_quat_to_blender`: a
    camera-to-world rotation has the camera's local axes as its columns, so
    relabelling which local axis is which is a right-multiplication. It happens
    to equal a +90 degree rotation about the camera's own X axis, but the matrix
    form is kept because it is literally the tested Python function's body.

    Positions need no conversion at all — both worlds are right-handed and Z-up.
    """
    w, x, y, z = quat_wxyz
    r_cpl = Quaternion((w, x, y, z)).normalized().to_matrix()
    return (r_cpl @ BLENDER_AXES_IN_CPL).to_quaternion()


def lens_from_fov(fov_degrees: float, sensor_width: float = SENSOR_WIDTH_MM) -> float:
    """Focal length in mm for a horizontal FOV, on a HORIZONTAL-fit sensor."""
    fov = min(max(fov_degrees, MIN_RENDERABLE_FOV), MAX_RENDERABLE_FOV)
    return sensor_width / (2.0 * math.tan(math.radians(fov) * 0.5))


# ---------------------------------------------------------------------------
# Cage fitting
# ---------------------------------------------------------------------------


@dataclass
class CageFit:
    """How the cage is positioned and sized for one particular trajectory."""

    centre: Vector          # lateral centre (x, y); z is `floor_z`
    floor_z: float
    scale: float            # multiplier on every cage dimension
    span: float             # bounding-box diagonal of the camera path
    path_length: float
    samples: list[Vector]   # points along the path, for near-field placement

    def unit(self, value: float) -> float:
        return value * self.scale


def fit_cage(poses: list[Pose], *, path_samples: int = 96) -> CageFit:
    """Size and place the cage around this trajectory.

    The cage is fitted rather than fixed because the trajectory's magnitude is
    not known in advance: normalized mode lands near `REFERENCE_SPAN`, but a
    metric calibration can produce a 400-unit dolly or a 0.3-unit macro move,
    and a fixed cage would give the first no near field and the second no
    parallax at all.
    """
    points = [Vector(p.position) for p in poses]
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    span = (hi - lo).length
    path_length = sum((points[i + 1] - points[i]).length for i in range(len(points) - 1))

    # A trajectory with no translation (a pure pan, or a static camera) carries
    # no information about world scale, so assume normalized units rather than
    # deriving a scale from noise.
    extent = span if span > 1e-4 else REFERENCE_SPAN
    scale = min(max(extent / REFERENCE_SPAN, MIN_CAGE_SCALE), MAX_CAGE_SCALE)

    step = max(1, len(points) // max(1, path_samples))
    samples = points[::step] or [points[0]]

    return CageFit(
        centre=Vector(((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, 0.0)),
        floor_z=lo.z - FLOOR_CLEARANCE * scale,
        scale=scale,
        span=span,
        path_length=path_length,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def flat_material(name: str, value: float) -> bpy.types.Material:
    """Matte monochrome diffuse.

    Diffuse rather than Principled on purpose: no specular highlight means no
    view-dependent shading, so a surface's brightness does not change as the
    camera moves past it. Anything that changes with view angle is a false
    motion cue in a clip whose only job is to carry motion.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (value, value, value, 1.0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _prototype_mesh(kind: str, **kwargs) -> bpy.types.Mesh:
    """Build one primitive and keep only its mesh datablock.

    Instances share the mesh, so a few hundred markers cost a few hundred object
    headers rather than a few hundred copies of the geometry — and the operator
    that builds primitives runs once instead of once per marker.
    """
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, vertices=16)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, subdivisions=2)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1.0, minor_radius=kwargs.get("minor", 0.06),
            major_segments=40, minor_segments=8,
        )
    elif kind == "plane":
        bpy.ops.mesh.primitive_plane_add(size=1.0)
    else:
        raise ValueError(f"unknown prototype '{kind}'")

    obj = bpy.context.active_object
    mesh = obj.data
    mesh.name = f"proto_{kind}"
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def _instance(
    mesh: bpy.types.Mesh, name: str, location: Vector, scale: Vector,
    material: bpy.types.Material, rotation: Euler | Quaternion | None = None,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.scale = scale
    if rotation is not None:
        if isinstance(rotation, Quaternion):
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = rotation
        else:
            obj.rotation_euler = rotation
    if mesh.materials:
        # Shared mesh: override the material per object rather than per mesh.
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = material
    else:
        mesh.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _grid_mesh(name: str, spacing: float, half_extent: float, width: float) -> bpy.types.Mesh:
    """One mesh holding every line of a square grid, as thin flat quads.

    Real geometry rather than a checker texture: a texture at grazing angles
    moires badly under motion, and a moire pattern that crawls as the camera
    moves is a false motion cue.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    half_width = width * 0.5
    count = int(half_extent / spacing)
    for i in range(-count, count + 1):
        centre = i * spacing
        for (x0, y0, x1, y1) in (
            (centre - half_width, -half_extent, centre + half_width, half_extent),
            (-half_extent, centre - half_width, half_extent, centre + half_width),
        ):
            base = len(verts)
            verts.extend([
                (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
            ])
            faces.append((base, base + 1, base + 2, base + 3))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


@dataclass
class StyleRecipe:
    """What a proxy style contains.

    Every style keeps near/mid/far anchors. A style without depth layering would
    make translation unreadable, which is the one thing this clip exists to show
    — so "minimal" means fewer objects, never fewer depths.
    """

    grid_spacing: float
    pole_rings: tuple[float, ...]
    path_markers: int
    blocks: int
    spheres: int
    gate_radii: tuple[float, ...]
    ladders: bool


STYLE_RECIPES = {
    "motion_cage": StyleRecipe(
        grid_spacing=2.0,
        pole_rings=(5.0, 11.0, 21.0, 38.0, 66.0, 112.0),
        path_markers=56, blocks=44, spheres=28,
        gate_radii=(8.0, 20.0, 45.0), ladders=True,
    ),
    "depth_poles": StyleRecipe(
        grid_spacing=4.0,
        pole_rings=(4.0, 7.0, 11.0, 17.0, 26.0, 40.0, 62.0, 96.0, 150.0),
        path_markers=64, blocks=0, spheres=0,
        gate_radii=(9.0, 26.0), ladders=True,
    ),
    "ground_grid": StyleRecipe(
        grid_spacing=1.5,
        pole_rings=(18.0, 48.0, 110.0),
        path_markers=18, blocks=10, spheres=6,
        gate_radii=(10.0, 30.0, 70.0), ladders=False,
    ),
    "minimal": StyleRecipe(
        grid_spacing=4.0,
        pole_rings=(7.0, 24.0, 70.0),
        path_markers=12, blocks=6, spheres=4,
        gate_radii=(9.0,), ladders=True,
    ),
}


def build_cage(style: str, fit: CageFit, *, seed: int = 11) -> int:
    """Build the motion cage. Returns the number of objects created."""
    recipe = STYLE_RECIPES[style]
    rng = random.Random(seed)
    u = fit.unit
    base = Vector((fit.centre.x, fit.centre.y, fit.floor_z))

    materials = {key: flat_material(f"cpl_{key}", value) for key, value in GREYS.items()}
    protos = {
        "cube": _prototype_mesh("cube"),
        "cylinder": _prototype_mesh("cylinder"),
        "sphere": _prototype_mesh("sphere"),
        "plane": _prototype_mesh("plane"),
    }
    count = 0

    # --- ground -------------------------------------------------------------
    ground_size = u(BACKDROP_RADIUS * 2.6)
    _instance(
        protos["plane"], "cpl_ground", base,
        Vector((ground_size, ground_size, 1.0)), materials["floor"],
    )
    count += 1

    # --- grid: two densities, so both fine and coarse motion are countable ---
    for tier, (spacing_mult, width, key) in enumerate((
        (1.0, 0.05, "grid_minor"),
        (5.0, 0.16, "grid_major"),
    )):
        spacing = u(recipe.grid_spacing * spacing_mult)
        mesh = _grid_mesh(
            f"cpl_grid_{tier}", spacing, u(BACKDROP_RADIUS * 0.62), u(width)
        )
        # Stacked by a hair so the major lines win the depth test over the minor
        # ones; coplanar quads z-fight and the flicker reads as motion.
        _instance(
            mesh, f"cpl_grid_{tier}",
            base + Vector((0.0, 0.0, u(0.01 + 0.01 * tier))),
            Vector((1.0, 1.0, 1.0)), materials[key],
        )
        count += 1

    # --- backdrop + horizon band -------------------------------------------
    shell_height = u(BACKDROP_RADIUS * 1.1)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=u(BACKDROP_RADIUS), depth=shell_height, vertices=96,
        end_fill_type="NOTHING",
        location=base + Vector((0.0, 0.0, shell_height * 0.5 - u(4.0))),
    )
    shell = bpy.context.active_object
    shell.name = "cpl_backdrop"
    shell.data.materials.append(materials["backdrop"])
    count += 1

    bpy.ops.mesh.primitive_cylinder_add(
        radius=u(BACKDROP_RADIUS * 0.995), depth=u(1.6), vertices=96,
        end_fill_type="NOTHING",
        location=base + Vector((0.0, 0.0, u(0.8))),
    )
    band = bpy.context.active_object
    band.name = "cpl_horizon_band"
    band.data.materials.append(materials["horizon"])
    count += 1

    # --- poles in concentric rings -----------------------------------------
    # Rings rather than a random cloud: they guarantee a near, a mid and a far
    # vertical in *every* direction, so the depth cue survives whichever way the
    # recovered camera happens to look.
    for ring_index, radius in enumerate(recipe.pole_rings):
        azimuths = 6 + ring_index * 4
        # Golden-angle offset per ring so poles never line up radially and hide
        # each other.
        offset = ring_index * 2.39996
        for k in range(azimuths):
            angle = offset + 2.0 * math.pi * k / azimuths
            height = u(3.0 + 8.0 * rng.random())
            thickness = u(0.09 + 0.13 * rng.random())
            _instance(
                protos["cylinder"], f"cpl_pole_{ring_index}_{k}",
                base + Vector((
                    math.cos(angle) * u(radius) * (0.9 + 0.2 * rng.random()),
                    math.sin(angle) * u(radius) * (0.9 + 0.2 * rng.random()),
                    height * 0.5,
                )),
                Vector((thickness, thickness, height)), materials["pole"],
            )
            count += 1

    # --- markers along the camera path --------------------------------------
    # Placed relative to sampled path points, not to the origin: a long travel
    # would otherwise leave the near field empty for most of the shot, and near
    # objects are what make translation visible at all.
    path_xy = [(sample.x, sample.y) for sample in fit.samples]

    def clear_of_path(x: float, y: float, half_extent: float) -> bool:
        need = u(CAMERA_CLEARANCE) + half_extent
        need_sq = need * need
        return all((x - px) ** 2 + (y - py) ** 2 >= need_sq for px, py in path_xy)

    def distance_to_path(x: float, y: float) -> float:
        return math.sqrt(min((x - px) ** 2 + (y - py) ** 2 for px, py in path_xy))

    def angular_shrink(x: float, y: float, half_extent: float) -> float:
        """Uniform scale factor (<= 1) that keeps the object's angular size capped."""
        limit = MAX_ANGULAR_HALF_EXTENT * distance_to_path(x, y)
        return 1.0 if half_extent <= limit or half_extent <= 0 else limit / half_extent

    def place(anchor_for_attempt, radius_for_attempt, half_extent: float):
        """Rejection-sample a floor position clear of the whole camera path."""
        for _ in range(PLACEMENT_ATTEMPTS):
            anchor = anchor_for_attempt()
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = radius_for_attempt()
            x = anchor.x + math.cos(angle) * radius
            y = anchor.y + math.sin(angle) * radius
            if clear_of_path(x, y, half_extent):
                return x, y
        return None

    skipped = 0
    for i in range(recipe.path_markers):
        height = u(2.0 + 9.0 * rng.random())
        thickness = u(0.08 + 0.14 * rng.random())
        spot = place(
            lambda: fit.samples[rng.randrange(len(fit.samples))],
            lambda: u(3.0 + 27.0 * rng.random() ** 1.6),
            thickness,
        )
        if spot is None:
            skipped += 1
            continue
        _instance(
            protos["cylinder"], f"cpl_path_pole_{i}",
            Vector((spot[0], spot[1], fit.floor_z + height * 0.5)),
            Vector((thickness, thickness, height)), materials["pole"],
        )
        count += 1

    # --- blocks and spheres, half near the path and half around the cage ----
    for i in range(recipe.blocks):
        size = Vector((
            u(0.6 + 2.6 * rng.random()), u(0.6 + 2.6 * rng.random()),
            u(0.6 + 3.4 * rng.random()),
        ))
        # Half-diagonal: the footprint is rotated arbitrarily about Z.
        half = 0.5 * math.hypot(size.x, size.y)
        near_path = i % 2 == 0
        spot = place(
            (lambda: fit.samples[rng.randrange(len(fit.samples))]) if near_path else (lambda: base),
            lambda: u(4.0 + 80.0 * rng.random() ** 1.8),
            half,
        )
        if spot is None:
            skipped += 1
            continue
        size = size * angular_shrink(spot[0], spot[1], max(half, size.z * 0.5))
        _instance(
            protos["cube"], f"cpl_block_{i}",
            Vector((spot[0], spot[1], fit.floor_z + size.z * 0.5)),
            size, materials["block"],
            rotation=Euler((0.0, 0.0, rng.uniform(0.0, math.pi))),
        )
        count += 1

    for i in range(recipe.spheres):
        size = u(0.7 + 2.0 * rng.random())
        near_path = i % 2 == 0
        spot = place(
            (lambda: fit.samples[rng.randrange(len(fit.samples))]) if near_path else (lambda: base),
            lambda: u(4.0 + 60.0 * rng.random() ** 1.6),
            size,
        )
        if spot is None:
            skipped += 1
            continue
        # The sphere prototype has radius 0.5, so `size` is a diameter.
        size = size * angular_shrink(spot[0], spot[1], size * 0.5)
        _instance(
            protos["sphere"], f"cpl_sphere_{i}",
            Vector((spot[0], spot[1], fit.floor_z + u(1.0 + 5.0 * rng.random()))),
            Vector((size, size, size)), materials["block"],
        )
        count += 1

    if skipped:
        print(f"[proxy] {skipped} scattered object(s) skipped: no spot clear of the camera path",
              flush=True)

    # --- depth gates: explicit near / mid / far anchors ---------------------
    # Rings standing on edge in four directions. Flying towards one, past one
    # and away from a third makes the sign and magnitude of translation obvious
    # in a way scattered geometry does not.
    depth_keys = ("near", "mid", "far")
    for depth_index, radius in enumerate(recipe.gate_radii):
        key = depth_keys[min(depth_index, len(depth_keys) - 1)]
        gate_mesh = _prototype_mesh("torus", minor=0.05)
        major = u(2.2 + 1.4 * depth_index)
        for k in range(4):
            angle = math.pi * 0.5 * k + math.pi * 0.25
            _instance(
                gate_mesh, f"cpl_gate_{depth_index}_{k}",
                base + Vector((
                    math.cos(angle) * u(radius), math.sin(angle) * u(radius),
                    u(FLOOR_CLEARANCE),
                )),
                Vector((major, major, major)), materials[key],
                # Torus lies in XY by default; stand it on edge and face the centre.
                rotation=Euler((math.pi * 0.5, 0.0, angle + math.pi * 0.5)),
            )
            count += 1

    # --- equal-angular-size depth ladders ----------------------------------
    if recipe.ladders:
        for direction in range(4):
            angle = math.pi * 0.5 * direction + math.pi * 0.125
            for step, distance in enumerate((4.0, 8.0, 16.0, 32.0, 64.0, 120.0)):
                # Radius proportional to distance, so every sphere subtends the
                # same angle on screen. Screen size therefore says nothing about
                # depth and only parallax does.
                size = u(distance * 0.055)
                key = depth_keys[min(step // 2, len(depth_keys) - 1)]
                _instance(
                    protos["sphere"], f"cpl_ladder_{direction}_{step}",
                    base + Vector((
                        math.cos(angle) * u(distance), math.sin(angle) * u(distance),
                        u(FLOOR_CLEARANCE) + size,
                    )),
                    Vector((size, size, size)), materials[key],
                )
                count += 1

    count += build_lighting(fit)
    return count


def build_lighting(fit: CageFit) -> int:
    """Flat, even, and static.

    One shadow-casting key gives surfaces contact with the ground, which is a
    genuine depth cue. The fills cast no shadows, so nothing in the frame is
    lost to darkness. Every light is static: a light that moved with the camera
    would paint brightness changes onto the geometry, and a motion-reference
    clip must not contain any change that is not motion.
    """
    specs = (
        ("key", 2.6, (math.radians(52.0), math.radians(12.0), math.radians(-35.0)), True),
        ("fill", 1.5, (math.radians(64.0), math.radians(-18.0), math.radians(115.0)), False),
        ("rim", 1.1, (math.radians(108.0), math.radians(8.0), math.radians(20.0)), False),
    )
    for name, energy, rotation, shadow in specs:
        data = bpy.data.lights.new(f"cpl_sun_{name}", type="SUN")
        data.energy = energy
        data.angle = 0.6  # wide angular size -> soft shadow edges
        if hasattr(data, "use_shadow"):
            data.use_shadow = shadow
        obj = bpy.data.objects.new(f"cpl_sun_{name}", data)
        obj.location = Vector((fit.centre.x, fit.centre.y, fit.floor_z + fit.unit(80.0)))
        obj.rotation_euler = Euler(rotation)
        bpy.context.scene.collection.objects.link(obj)

    world = bpy.data.worlds.new("cpl_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        # Light grey sky: an ambient term so nothing is black, and a value the
        # horizon band reads clearly against.
        background.inputs[0].default_value = (0.62, 0.63, 0.65, 1.0)
        background.inputs[1].default_value = 1.0
    return len(specs)


# ---------------------------------------------------------------------------
# Camera animation
# ---------------------------------------------------------------------------


def iter_fcurves(action):
    """Yield every F-curve in an action, across Blender API generations.

    Blender 4.4 replaced the flat `action.fcurves` list with layered, slotted
    actions (`action.layers[].strips[].channelbags[].fcurves`). Both spellings
    are handled so this runs on current and older Blender alike.
    """
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                for bag in getattr(strip, "channelbags", []) or []:
                    yield from getattr(bag, "fcurves", []) or []
        return
    yield from getattr(action, "fcurves", []) or []


def force_linear_interpolation(*owners) -> None:
    """Make every keyframe LINEAR.

    Load-bearing for invariants I1 and I10, not cosmetic. Keys land on every
    rendered frame, so the values at the sample points are already exact — but
    Blender's default Bezier handles ease in and out of every key, which changes
    the motion *between* samples. That is what motion blur integrates over, what
    a user scrubbing the exported .blend sees, and what any re-render at a
    different frame rate would sample. An "accelerating" move must not be eased
    into something else on the way out.
    """
    for owner in owners:
        anim = getattr(owner, "animation_data", None)
        if not anim or not anim.action:
            continue
        for fcurve in iter_fcurves(anim.action):
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def animate_camera(trajectory: Trajectory, fit: CageFit) -> tuple[bpy.types.Object, list[dict]]:
    """Create the camera and key the recovered motion onto it.

    Returns the camera plus the poses as actually keyed, in Blender convention —
    the same records `gen_synthetic_scene.py` writes as ground truth, so the
    proxy can be fed back through the validation harness.
    """
    cam_data = bpy.data.cameras.new("cpl_camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    # The backdrop sits at BACKDROP_RADIUS cage units and the camera may be at
    # the far side of its own path, so the far clip has to cover both. Blender's
    # 100 m default would cut the whole far field away.
    cam_data.clip_start = max(fit.unit(0.02), 1e-4)
    cam_data.clip_end = fit.unit(BACKDROP_RADIUS * 3.0) + fit.span * 2.0

    cam = bpy.data.objects.new("cpl_camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"

    keyed: list[dict] = []
    for index, pose in enumerate(trajectory.poses):
        frame = index + 1
        quat = camerapath_quat_to_blender(pose.quaternion)
        lens = lens_from_fov(pose.fov)

        cam.location = Vector(pose.position)
        cam.rotation_quaternion = quat
        cam_data.lens = lens

        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_quaternion", frame=frame)
        cam_data.keyframe_insert("lens", frame=frame)

        keyed.append({
            "frame": pose.frame,
            "scene_frame": frame,
            "time": pose.time,
            "location": list(pose.position),
            "quaternion_wxyz": [quat.w, quat.x, quat.y, quat.z],
            "lens_mm": lens,
            "sensor_width_mm": SENSOR_WIDTH_MM,
            "fov_horizontal": pose.fov,
        })

    force_linear_interpolation(cam, cam_data)
    return cam, keyed


def report_timing(trajectory: Trajectory) -> None:
    """State, on stdout, how the output clock relates to the source clock (I1).

    Blender renders at a constant rate. A VFR source has non-uniform frame
    times, so the per-frame times are resampled onto a uniform grid while the
    total duration is preserved. That is a real (if small) change and it gets
    said out loud rather than hidden.
    """
    times = [p.time for p in trajectory.poses if math.isfinite(p.time)]
    emitted = trajectory.duration
    print(
        f"[proxy] timing: {len(trajectory.poses)} frames at {trajectory.fps:.6f} fps "
        f"-> {emitted:.6f}s output",
        flush=True,
    )
    if len(times) < 2:
        print("[proxy] timing: frames carry no timestamps; trusting declared fps", flush=True)
        return

    deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    positive = [d for d in deltas if d > 1e-9]
    if not positive:
        return
    mean_dt = sum(positive) / len(positive)
    variance = sum((d - mean_dt) ** 2 for d in positive) / len(positive)
    jitter = (variance ** 0.5) / mean_dt
    measured = times[-1] - times[0] + mean_dt
    drift = abs(measured - emitted)

    print(
        f"[proxy] timing: source span {measured:.6f}s, jitter {jitter * 100:.2f}%, "
        f"drift {drift * 1000.0:.2f}ms",
        flush=True,
    )
    if jitter > TIMING_JITTER_WARN:
        print(
            f"[proxy] WARNING: source frame timing is non-uniform ({jitter * 100:.1f}%); "
            "output is constant-rate, so per-frame times are resampled onto a "
            "uniform grid. Total duration is preserved.",
            flush=True,
        )
    if drift > 1.0 / trajectory.fps:
        print(
            f"[proxy] WARNING: emitted duration differs from the source span by "
            f"{drift * 1000.0:.1f}ms, more than one output frame "
            f"({1000.0 / trajectory.fps:.1f}ms) — invariant I1 is at risk. "
            "The trajectory's fps and frame times disagree.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------


def available_render_engines() -> list[str]:
    """Engine identifiers this Blender build actually offers.

    Read from the RNA enum rather than hard-coded: the identifier has changed
    across releases (EEVEE -> BLENDER_EEVEE_NEXT -> BLENDER_EEVEE again) and is
    also affected by which add-ons are enabled, so assigning a name that does
    not exist raises at render time.
    """
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        return [item.identifier for item in prop.enum_items]
    except Exception:  # noqa: BLE001
        return ["BLENDER_WORKBENCH"]


def pick_render_engine() -> str:
    """EEVEE if this build has it; Workbench as the fallback.

    Cycles would be pointlessly slow for flat diffuse geometry. Workbench is a
    genuine fallback rather than an equal: its matcap shading flattens the
    depth cues that the cage is built to provide.
    """
    engines = available_render_engines()
    for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if candidate in engines:
            return candidate
    return engines[0] if engines else "BLENDER_WORKBENCH"


def configure_render(
    *, width: int, height: int, frames: int, fps: float, stem: str, samples: int,
) -> tuple[int, float]:
    """Set up video output. Returns the (fps, fps_base) pair actually applied."""
    scene = bpy.context.scene
    scene.render.engine = pick_render_engine()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    fps_num, fps_base = scene_fps_pair(fps)
    scene.render.fps = fps_num
    scene.render.fps_base = fps_base
    scene.frame_start = 1
    scene.frame_end = frames
    scene.frame_step = 1

    # Blender 5 gates the video formats behind media_type: on that build
    # file_format="FFMPEG" is rejected outright until media_type is VIDEO.
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = 12
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = stem
    scene.render.use_overwrite = True
    scene.render.film_transparent = False

    # Motion blur off: it would smear the per-frame pose that is the whole
    # payload, and it costs render time to do it.
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False

    # A display transform would tone-map the grey palette into something else.
    # "Standard" keeps the authored values; the name depends on the OCIO config,
    # so fall back rather than fail.
    for transform in ("Standard", "Raw", "NONE"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue

    if scene.render.engine.startswith("BLENDER_EEVEE"):
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass

    return fps_num, fps_base


def collect_rendered_file(directory: str, stem_name: str) -> str | None:
    """Find what the FFMPEG writer actually produced.

    `bpy.ops.render.render(animation=True)` appends the rendered frame range to
    the filename, so the requested path is never the produced path.
    """
    candidates = [
        name for name in os.listdir(directory)
        if name.startswith(stem_name) and name.lower().endswith(".mp4")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda n: os.path.getmtime(os.path.join(directory, n)))
    return os.path.join(directory, candidates[-1])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="render_motion_proxy",
        description="Render a recovered camera trajectory as a neutral motion proxy MP4.",
    )
    parser.add_argument("--trajectory", required=True, help="trajectory JSON (spec §19)")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--style", default="motion_cage", choices=STYLES)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--fps", type=float, default=None,
        help="override the trajectory's fps; omit to preserve source timing (I1)",
    )
    parser.add_argument("--samples", type=int, default=16, help="EEVEE TAA render samples")
    parser.add_argument("--seed", type=int, default=11, help="cage layout seed")
    parser.add_argument("--blend", default=None, help="also save the scene as a .blend")
    parser.add_argument(
        "--dump-poses", default=None,
        help="write the poses as keyed, in Blender convention, to this JSON file",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="build the scene (and .blend) without rendering",
    )
    return parser.parse_args(argv)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    trajectory = load_trajectory(args.trajectory)
    if args.fps is not None:
        if args.fps <= 0.0:
            raise TrajectoryError(f"--fps must be positive, got {args.fps}")
        trajectory.fps = args.fps

    frames = len(trajectory.poses)
    print(
        f"[proxy] loaded {frames} poses from {os.path.basename(args.trajectory)} "
        f"({trajectory.shot_count} shot(s), scale_mode={trajectory.scale_mode})",
        flush=True,
    )
    report_timing(trajectory)

    clear_scene()
    fit = fit_cage(trajectory.poses)
    objects = build_cage(args.style, fit, seed=args.seed)
    print(
        f"[proxy] cage style={args.style} objects={objects} scale={fit.scale:.4f} "
        f"span={fit.span:.4f} path_length={fit.path_length:.4f} floor_z={fit.floor_z:.4f}",
        flush=True,
    )

    _, keyed = animate_camera(trajectory, fit)

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem_name = os.path.splitext(os.path.basename(out_path))[0] + "_render_"

    fps_num, fps_base = configure_render(
        width=args.width, height=args.height, frames=frames, fps=trajectory.fps,
        stem=os.path.join(out_dir, stem_name), samples=args.samples,
    )

    if args.dump_poses:
        with open(args.dump_poses, "w", encoding="utf-8") as fh:
            json.dump({
                "convention": {
                    "note": (
                        "Poses as keyed into Blender: right-handed Z-up world, camera "
                        "looks along local -Z with local +Y up, quaternions [w,x,y,z] "
                        "camera-to-world. Produced by camerapath_quat_to_blender(), "
                        "which mirrors app.geometry.conventions."
                    ),
                    "world_up": "+Z",
                    "camera_forward": "-Z",
                    "camera_up": "+Y",
                    "quaternion_order": "wxyz",
                },
                "fps": trajectory.fps,
                "fps_numerator": fps_num,
                "fps_base": fps_base,
                "frame_count": frames,
                "frames": keyed,
            }, fh, indent=1)
        print(f"[proxy] wrote keyed poses to {args.dump_poses}", flush=True)

    rendered: str | None = None
    if args.no_render:
        print("[proxy] --no-render: scene built, skipping the render", flush=True)
    else:
        bpy.ops.render.render(animation=True)
        rendered = collect_rendered_file(out_dir, stem_name)
        if rendered is None:
            raise RuntimeError(
                f"render produced no MP4 matching '{stem_name}*' in {out_dir}"
            )
        if os.path.abspath(rendered) != out_path:
            os.replace(rendered, out_path)
        print(f"[proxy] wrote {out_path}", flush=True)

    if args.blend:
        blend_path = os.path.abspath(args.blend)
        os.makedirs(os.path.dirname(blend_path) or ".", exist_ok=True)
        # Point the saved scene at the real output path, so opening the .blend
        # and hitting render reproduces the same file.
        bpy.context.scene.render.filepath = os.path.splitext(out_path)[0]
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"[proxy] saved {blend_path}", flush=True)

    # Machine-readable summary. `blender/runner.py` parses this line and then
    # re-verifies the numbers against the file with ffprobe — Blender reporting
    # the right frame count is necessary but not sufficient (I1).
    print(
        "CPL_RESULT "
        f"frames={frames} fps={fps_num / fps_base:.6f} fps_num={fps_num} "
        f"fps_base={fps_base:.9f} width={args.width} height={args.height} "
        f"duration={frames * fps_base / fps_num:.6f} style={args.style} "
        f"rendered={'0' if args.no_render else '1'} out={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
