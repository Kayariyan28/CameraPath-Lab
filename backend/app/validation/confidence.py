"""Evidence-weighted confidence for one recovered shot trajectory (spec §22).

Invariant I7: confidence is *computed from evidence* and is allowed to be LOW.
The reason that invariant exists is that this product's failure cases are
invisible to the user. A trajectory that is mirrored, that has an invented
baseline, or that is a screen-space match dressed up as a camera path all look
exactly like a correct result in the viewer. The only defence is a verdict that
is honest about what the footage actually supports, and that says so in language
a director can act on.

Four consequences shape this module:

1. **Three separate confidences.** Translation, rotation and zoom are recovered
   from *different* evidence and genuinely differ in quality. A tripod pan across
   a distant skyline has excellent rotation evidence (thousands of tracks moving
   coherently) and *no* translation evidence whatsoever — the flow is explained
   by a homography, so no baseline is observable. Collapsing that into one number
   would either slander the rotation or launder the translation. So the report
   carries all three, and the aggregate `level` is a verdict on the *physical
   trajectory*, capped by the weakest thing that trajectory depends on.

2. **Missing evidence is omitted, never filled in.** A backend that reports no
   reprojection error contributes no reprojection term; the remaining weights are
   renormalised and a reason records the gap. Substituting a "neutral" 0.5 would
   be a fabricated measurement, which is exactly what I7 forbids. The same rule
   applies to things that *look* like evidence but are not: a focal length the
   solve could not constrain is a prior, and a flat prior curve is not "stable".

3. **Hard gates, not just a score.** HIGH requires specific, individually
   checkable conditions (translation observable, a physical solve, frames
   actually registered, a real parallax baseline). A weighted score alone can
   reach HIGH by piling up cheap evidence — texture, sharpness, tracking quality
   — on a shot whose translation is fundamentally unobservable. The gates make
   that impossible.

4. **Discontinuities are judged against the footage, not against smoothness.**
   Handheld jitter is the product (I10), so a jumpy trajectory is only penalised
   where the jump is *unsupported*: a pose step far larger than its neighbours
   while the measured image motion at the same moment shows no matching spike. A
   real bump in the source moves the image too, and survives unpenalised.

The geometric cross-check — do two backends agree on the *path*? — is computed
by `validation/metrics.compare_solvers`, because it needs both trajectories. When
the caller has run it, its result can be passed in and `solver_agreement` is
path agreement; otherwise it is agreement between the backends' recorded
*verdicts*, weighted lower and labelled as such. Either way it is None unless at
least two backends succeeded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.core.logging import get_logger
from app.geometry.intrinsics import focal_pixels_to_fov, fov_to_focal_pixels
from app.geometry.rotations import quat_angular_distance
from app.models.schemas.motion import (
    LensFrame,
    MotionFrame,
    MotionModel,
    MotionSignature,
)
from app.models.schemas.trajectory import (
    CameraPose,
    ConfidenceLevel,
    ConfidenceReport,
    PipelineMode,
    SolverDecision,
    SolverSource,
)
from app.solvers.base import GeometryResult

log = get_logger("validation.confidence")

# --------------------------------------------------------------------------
# Thresholds. Every one of these is a judgement call, so each is named and
# justified rather than inlined as a magic number.
# --------------------------------------------------------------------------

#: Parallax score below which depth-dependent flow is indistinguishable from
#: tracking noise, so translation is not observable. Deliberately the same
#: threshold `tracking.parallax.summarize` and the AUTO router use — two
#: different cut-offs for the same question would let the modules disagree.
PARALLAX_OBSERVABLE_MIN = 0.15

#: Parallax at which translation evidence is unambiguous and the term saturates.
PARALLAX_STRONG = 0.60

#: Hard ceiling on `translation_confidence` when translation is not observable.
#: Not exactly zero, because the field is a confidence and not a measurement,
#: but small enough that any consumer thresholding above 0.1 rejects it.
UNOBSERVABLE_TRANSLATION_CEILING = 0.05

#: Ceiling on `translation_confidence` for a perceptual solve. A screen-space
#: match can be excellent *as a match* while saying nothing about the physical
#: path, so its translation never scores above "weak".
PERCEPTUAL_TRANSLATION_CEILING = 0.35

#: Reprojection error (px at the solve's image resolution) that counts as an
#: excellent bundle vs. one carrying no useful geometry. Sub-pixel is the mark of
#: a converged solve; beyond ~4 px the tracks and the structure disagree by more
#: than a feature detector's own localisation error.
REPROJECTION_EXCELLENT_PX = 0.75
REPROJECTION_POOR_PX = 4.0

#: Bundle-adjustment residual bounds, same units and same reasoning.
BA_RESIDUAL_EXCELLENT = 0.50
BA_RESIDUAL_POOR = 3.00

#: Inlier ratio of the dense 2D model: below this the "dominant motion" is not
#: dominant, above it the model explains the frame.
INLIER_POOR = 0.30
INLIER_GOOD = 0.80

#: Per-transition inlier ratio at which a transition counts as fully measured.
MEASUREMENT_GOOD_INLIER = 0.60

#: Registered-frame ratios bracketing "barely anything solved" and "essentially
#: everything solved".
REGISTERED_POOR = 0.20
REGISTERED_GOOD = 0.90

#: Track-count bracket. COLMAP on a textured shot yields thousands of 3D points;
#: below ~50 the structure is too thin to constrain a bundle.
TRACKS_POOR = 50
TRACKS_GOOD = 800

#: Texture and sharpness brackets (both scores are already 0-1; sharpness is the
#: `MotionSignature.blur_score` convention, 1 = sharp).
TEXTURE_POOR = 0.15
TEXTURE_GOOD = 0.60
SHARPNESS_POOR = 0.20
SHARPNESS_GOOD = 0.70

#: Median flow (px) below which the camera is effectively static. The same
#: figure the AUTO router uses to route a static shot away from SfM.
STATIC_FLOW_PX = 0.4

#: Discontinuity detection. A step is judged against the median of its
#: +-DISCONTINUITY_WINDOW neighbours rather than the whole shot, so a camera that
#: holds still and then pans is not flagged (a global median would be ~0 and every
#: moving step would look like a jump). Uniform handheld jitter raises the local
#: median with it: for Rayleigh-distributed step noise, exceeding 3x the median
#: plus the floor is a ~4 sigma event, so honest jitter essentially never trips it.
DISCONTINUITY_WINDOW = 4
STEP_OUTLIER_FACTOR = 3.0

#: Floor added to the threshold, as a fraction of the shot's mean step, so a step
#: next to perfectly still neighbours needs to be a real fraction of the shot's
#: typical motion before it counts.
STEP_FLOOR_FRACTION = 0.5

#: Absolute floors. 0.05 deg per transition is ~1 px at a 1500 px focal length:
#: below that a "jump" is not visible in any output. Positions are in scale-mode
#: units whose normalized paths span ~10-25 units, so 1e-9 only absorbs float noise.
ROTATION_STEP_FLOOR_DEG = 0.05
POSITION_STEP_FLOOR = 1e-9

#: Fewest steps a discontinuity judgement is made from. With fewer, a "local
#: neighbourhood" is a single other value.
MIN_CONTINUITY_STEPS = 4

#: A flagged pose step is *supported by the footage* when the measured image
#: motion over the same transition is at least this multiple of its own local
#: median. Deliberately looser than STEP_OUTLIER_FACTOR: the image signal is the
#: witness, not the defendant, and needs only to show that something happened.
IMAGE_SUPPORT_RATIO = 2.0

#: Fraction of unsupported discontinuous steps that drives temporal consistency
#: to zero.
OUTLIER_FRACTION_TOLERANCE = 0.10

#: RMS second difference of the focal curve, as a fraction of mean focal, that
#: drives focal stability to zero. A second difference is used rather than a
#: plain spread so a deliberate smooth zoom — which is *signal* — scores as
#: stable while a rattling focal estimate does not.
FOCAL_NOISE_TOLERANCE = 0.02

#: Ceiling on focal stability and zoom confidence when the solve reports that the
#: focal length is not observable. A flat prior curve would otherwise score as
#: perfectly stable — the most flattering possible number for something that was
#: never measured. Below MEDIUM_SCORE_MIN, so it always reads as LOW.
FOCAL_UNOBSERVED_CEILING = 0.20

#: The +-15% pin the solvers use to probe focal observability. Also the tolerance
#: for agreement between the solve's focal and the lens curve: a disagreement
#: larger than the probe is outside anything the solve considered.
FOCAL_PROBE_FRACTION = 0.15

#: Spread between two backends' recorded confidences / registration ratios that
#: counts as total disagreement, and the relative reprojection spread likewise.
SOLVER_VERDICT_TOLERANCE = 0.25
SOLVER_REPROJECTION_TOLERANCE = 0.50

#: Weight of solver agreement in the aggregate. Verdict agreement is a proxy
#: (two backends can report the same quality for different paths), so it counts
#: for half as much as a measured path comparison.
SOLVER_AGREEMENT_WEIGHT = 0.06
VERDICT_AGREEMENT_WEIGHT_FACTOR = 0.5

#: Radial-flow / scale-change brackets for "is there a zoom-like signal at all".
SCALE_CHANGE_NEGLIGIBLE = 0.02
SCALE_CHANGE_STRONG = 0.15
RADIAL_FLOW_NEGLIGIBLE_PX = 0.25
RADIAL_FLOW_STRONG_PX = 2.00

#: Parallax needed to separate a zoom from a dolly. Below it, an expanding image
#: is equally well explained by either, and the recovered FOV curve reproduces
#: the framing without establishing its physical cause.
ZOOM_SEPARABLE_PARALLAX = 0.55
ZOOM_CONFOUND_PENALTY = 0.50

#: Signature degeneracy wording thresholds.
PURE_ROTATION_DOMINANCE = 0.85
MOSTLY_ROTATION_DOMINANCE = 0.50

#: Aggregate score bands.
HIGH_SCORE_MIN = 0.72
MEDIUM_SCORE_MIN = 0.45

#: Additional hard gates for HIGH (all must hold).
HIGH_REGISTERED_RATIO_MIN = 0.60
HIGH_INLIER_RATIO_MIN = 0.50
HIGH_PARALLAX_MIN = 0.35
HIGH_REPROJECTION_MAX_PX = 2.50
HIGH_TEMPORAL_CONSISTENCY_MIN = 0.50
HIGH_TRACK_COUNT_MIN = 2 * TRACKS_POOR
HIGH_SOLVER_AGREEMENT_MIN = 0.50

#: Caps applied after scoring. Each encodes "no matter how good the rest of the
#: evidence is, the physical trajectory cannot be trusted more than X".
#:  * unobservable translation: below MEDIUM. The trajectory's positions carry no
#:    measurement, so as a physical reconstruction it is LOW (spec §22's own
#:    example); rotation quality is reported in `rotation_confidence` instead.
#:  * perceptual: below HIGH. It may be an excellent match; it is not a survey.
#:  * no reconstruction at all: firmly LOW.
UNOBSERVABLE_SCORE_CAP = 0.40
PERCEPTUAL_PHYSICAL_SCORE_CAP = 0.60
NO_GEOMETRY_SCORE_CAP = 0.35

#: Backends that triangulate structure. Only these register frames, produce a
#: genuine reprojection error, or can support a HIGH physical verdict.
GEOMETRIC_SOURCES = frozenset({
    SolverSource.OPENCV, SolverSource.COLMAP, SolverSource.VGGT, SolverSource.FUSED,
})

#: Reason priorities. The UI shows the list in order, so the decisive facts have
#: to come first — a user who reads one line should read the one that explains
#: the verdict.
_DECISIVE, _MODE, _PRIMARY, _SECONDARY = 0, 1, 2, 3


class TranslationVerdict(str, Enum):
    """Why translation is or is not observable. The cause changes what the user
    should do about it, so the reasons must not conflate them."""

    OBSERVABLE = "observable"
    NO_PARALLAX = "no_parallax"
    """The footage itself carries no depth-dependent flow."""
    NO_RECONSTRUCTION = "no_reconstruction"
    """The footage might, but no backend produced a solve to measure it."""
    SOLVER_REPORTED = "solver_reported"
    """Parallax is present but the selected solve did not recover a baseline."""


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _ramp(value: float, low: float, high: float) -> float:
    """0 at or below `low`, 1 at or above `high`, linear between."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return _clamp01((value - low) / (high - low))


def _falling_ramp(value: float, good: float, bad: float) -> float:
    """1 at or below `good`, 0 at or above `bad` — for error-like quantities."""
    if bad <= good:
        return 1.0 if value <= good else 0.0
    return _clamp01((bad - value) / (bad - good))


def _weighted_mean(terms: Sequence[tuple[float, float | None]]) -> float:
    """Weighted mean over the terms that actually have evidence.

    A `None` term is dropped and the remaining weights renormalised. Filling it
    with an invented neutral value would manufacture evidence (I7); dropping it
    means the missing measurement neither helps nor hurts, and the reasons record
    that it was missing.
    """
    total = 0.0
    weight = 0.0
    for w, value in terms:
        if value is None:
            continue
        total += w * _clamp01(value)
        weight += w
    return total / weight if weight > 0.0 else 0.0


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _band(score: float) -> ConfidenceLevel:
    if score >= HIGH_SCORE_MIN:
        return ConfidenceLevel.HIGH
    if score >= MEDIUM_SCORE_MIN:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# --------------------------------------------------------------------------
# Temporal discontinuities (shared with validation/metrics.py)
# --------------------------------------------------------------------------


def _local_medians(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Median of each element's finite neighbours within the window, excluding
    the element itself (a spike must not vouch for itself)."""
    n = len(values)
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - DISCONTINUITY_WINDOW), min(n, i + DISCONTINUITY_WINDOW + 1)
        idx = np.r_[lo:i, i + 1:hi]
        neighbours = values[idx][finite[idx]]
        out[i] = float(np.median(neighbours)) if len(neighbours) else 0.0
    return out


def discontinuity_flags(steps: np.ndarray, *, absolute_floor: float) -> np.ndarray | None:
    """Boolean mask of steps that are discontinuities relative to their
    neighbourhood. None when there are too few steps to judge.

    A non-finite step is always flagged: a pose that is NaN is broken, not quiet.
    """
    values = np.asarray(steps, dtype=np.float64).reshape(-1)
    if len(values) < MIN_CONTINUITY_STEPS:
        return None
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones(len(values), dtype=bool)
    mean_step = float(np.mean(np.abs(values[finite])))
    floor = max(absolute_floor, STEP_FLOOR_FRACTION * mean_step)
    local = _local_medians(values, finite)
    flags = ~finite
    flags |= finite & (np.where(finite, values, 0.0) > STEP_OUTLIER_FACTOR * local + floor)
    return flags


def angular_step_degrees(poses: Sequence[CameraPose]) -> np.ndarray:
    """Per-transition orientation change, in degrees.

    Measured as a quaternion angular distance, never by differencing Euler
    angles (invariant I3): Euler differences wrap, gimbal-lock and would report
    a 1 deg roll near the pole as a 180 deg jump.
    """
    if len(poses) < 2:
        return np.zeros(0)
    quats = [np.asarray(p.quaternion, dtype=np.float64) for p in poses]
    return np.array(
        [np.degrees(quat_angular_distance(quats[i - 1], quats[i])) for i in range(1, len(quats))]
    )


def position_step_lengths(poses: Sequence[CameraPose]) -> np.ndarray:
    """Per-transition translation magnitude, in scale-mode units (I5)."""
    if len(poses) < 2:
        return np.zeros(0)
    positions = np.array([p.position for p in poses], dtype=np.float64)
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def image_motion_steps(
    frame_indices: Sequence[int], motion_frames: Sequence[MotionFrame]
) -> np.ndarray | None:
    """Measured image motion (px) spanned by each consecutive pose pair.

    Sums the median flow of every transition in (f_a, f_b], so sparse anchor
    poses and dense per-frame poses are handled alike. A pose step no measured
    transition covers is NaN — absent evidence, not zero motion.

    This is image-space motion (I6). It is used purely as a *timing witness* —
    did something happen on screen at this moment? — and never as a measure of
    how far the camera moved.
    """
    if len(frame_indices) < 2 or not motion_frames:
        return None
    ordered = sorted(motion_frames, key=lambda mf: mf.frame_index)
    idx = np.array([mf.frame_index for mf in ordered], dtype=np.int64)
    flow = np.array([max(mf.flow_magnitude, 0.0) for mf in ordered], dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(flow)])
    steps = np.full(len(frame_indices) - 1, np.nan)
    for k in range(1, len(frame_indices)):
        a = int(np.searchsorted(idx, frame_indices[k - 1], side="right"))
        b = int(np.searchsorted(idx, frame_indices[k], side="right"))
        if b > a:
            steps[k - 1] = cumulative[b] - cumulative[a]
    return steps


def unsupported_discontinuity_fraction(
    steps: np.ndarray,
    frame_indices: Sequence[int],
    motion_frames: Sequence[MotionFrame],
    *,
    absolute_floor: float,
) -> float | None:
    """Fraction of pose steps that jump while the footage does not.

    The single definition of "a jump the source does not explain", shared with
    `validation/metrics.py`. A flagged step is excused when the measured image
    motion over the same transition also stands out from its neighbourhood: a
    real bump in the source moved the picture, and preserving it is the product
    (I10). Without image evidence nothing is excused.
    """
    flags = discontinuity_flags(steps, absolute_floor=absolute_floor)
    if flags is None:
        return None
    image = image_motion_steps(frame_indices, motion_frames)
    if image is not None and len(image) == len(flags):
        finite = np.isfinite(image)
        local = _local_medians(np.where(finite, image, 0.0), finite)
        supported = finite & (image > IMAGE_SUPPORT_RATIO * local) & (image > STATIC_FLOW_PX)
        flags = flags & ~supported
    return float(flags.mean())


# --------------------------------------------------------------------------
# Evidence extraction
# --------------------------------------------------------------------------


def _registration(geometry: GeometryResult, geometric: bool, poses: Sequence[CameraPose]
                  ) -> tuple[float, int, int]:
    """(ratio, registered, total) of *geometrically registered* frames.

    Strictly geometric: a perceptual or 2D-proxy solve produces a pose for every
    frame, and counting those here would present screen-space coverage as
    reconstruction coverage.
    """
    total = geometry.total_frames or len(poses)
    if not geometric:
        return 0.0, 0, total
    registered = geometry.registered_frames or geometry.pose_count
    if total <= 0:
        return 0.0, registered, 0
    return _clamp01(registered / total), registered, total


def _persistent_tracks(
    geometry: GeometryResult, geometric: bool, motion_frames: Sequence[MotionFrame]
) -> tuple[int, str]:
    """Tracks supporting the result, and what kind they are: triangulated 3D
    points when there is a reconstruction, otherwise the dense tracker's
    surviving background tracks per transition."""
    if geometric and geometry.track_count > 0:
        return int(geometry.track_count), "3d_points"
    if motion_frames:
        counts = [
            float(mf.background_track_count or mf.tracks_survived) for mf in motion_frames
        ]
        return int(np.median(counts)), "dense_tracks"
    return 0, "none"


def _mean_inlier_ratio(motion_frames: Sequence[MotionFrame], signature: MotionSignature) -> float:
    if motion_frames:
        return _clamp01(float(np.mean([mf.inlier_ratio for mf in motion_frames])))
    return _clamp01(signature.mean_inlier_ratio)


def _explicit_or_signature(explicit: float, recorded: float) -> float:
    """The caller's measurement, falling back to the value recorded on the shot
    signature only when the caller's is not a number.

    A measured 0.0 is kept. Treating zero as "missing" and substituting the
    signature's value would turn "no parallax" or "fully blurred" into whatever
    the analysis stage happened to record — optimistic by construction.
    """
    if explicit is not None and np.isfinite(explicit):
        return _clamp01(explicit)
    return _clamp01(recorded)


def _focal_stability(lens: Sequence[LensFrame]) -> float | None:
    """How *noisy* the focal curve is — not how much it changes.

    A deliberate zoom is signal and must not be punished, so the measure is the
    RMS second difference of the focal curve: zero for any smooth ramp, large for
    an estimate that rattles frame to frame. None with fewer than three samples,
    where a ramp cannot be told from noise.
    """
    if len(lens) < 3:
        return None
    focal = np.array([f.focal_normalized for f in lens], dtype=np.float64)
    if not np.isfinite(focal).all():
        return 0.0
    mean_focal = float(np.mean(np.abs(focal)))
    if mean_focal <= 1e-9:
        return 0.0
    second = focal[2:] - 2.0 * focal[1:-1] + focal[:-2]
    noise = float(np.sqrt(np.mean(second ** 2))) / mean_focal
    return _clamp01(1.0 - noise / FOCAL_NOISE_TOLERANCE)


@dataclass
class _FocalCheck:
    agreement: float
    geometry_fov: float
    lens_fov: float


def _geometry_fov(geometry: GeometryResult) -> float | None:
    """Horizontal FOV of the solve's focal length, or None when it cannot be
    stated. Focal in pixels is converted only through the width it refers to;
    the solve runs at a different resolution from the flow analysis."""
    if (
        geometry.focal_pixels is None
        or not np.isfinite(geometry.focal_pixels)
        or geometry.focal_pixels <= 0
        or not geometry.focal_image_width
    ):
        return None
    return focal_pixels_to_fov(geometry.focal_pixels, int(geometry.focal_image_width))


def _focal_agreement(
    geometry: GeometryResult, geometric: bool, lens: Sequence[LensFrame]
) -> _FocalCheck | None:
    """Does the lens curve the output will use match the focal the solve measured?

    Only meaningful when the solve actually measured focal. Compared as focal
    length ratio (resolution-free via FOV), against the solvers' +-15% probe.
    """
    if not geometric or not geometry.focal_observable or not lens:
        return None
    geometry_fov = _geometry_fov(geometry)
    if geometry_fov is None:
        return None
    lens_fov = float(np.median([f.fov_horizontal for f in lens]))
    # Any common width works: focal ratio at a fixed width is the FOV ratio.
    reference_width = 1000
    ratio = fov_to_focal_pixels(lens_fov, reference_width) / fov_to_focal_pixels(
        geometry_fov, reference_width
    )
    agreement = _clamp01(1.0 - abs(ratio - 1.0) / FOCAL_PROBE_FRACTION)
    return _FocalCheck(agreement=agreement, geometry_fov=geometry_fov, lens_fov=lens_fov)


def _pose_continuity(
    poses: Sequence[CameraPose], motion_frames: Sequence[MotionFrame]
) -> tuple[float | None, float | None, float | None]:
    """(continuity term, unsupported rotation fraction, unsupported position
    fraction). Continuity is None when there are too few poses to judge."""
    indices = [p.frame_index for p in poses]
    rotation = unsupported_discontinuity_fraction(
        angular_step_degrees(poses), indices, motion_frames,
        absolute_floor=ROTATION_STEP_FLOOR_DEG,
    )
    position = unsupported_discontinuity_fraction(
        position_step_lengths(poses), indices, motion_frames,
        absolute_floor=POSITION_STEP_FLOOR,
    )
    if rotation is None or position is None:
        return None, rotation, position
    worst = max(rotation, position)
    return _clamp01(1.0 - worst / OUTLIER_FRACTION_TOLERANCE), rotation, position


def _measurement_continuity(motion_frames: Sequence[MotionFrame]) -> float | None:
    """How much of the shot the dense 2D measurement actually explained."""
    if not motion_frames:
        return None
    per_frame = [
        0.0
        if mf.model_used is MotionModel.NONE
        else _clamp01(mf.inlier_ratio / MEASUREMENT_GOOD_INLIER)
        for mf in motion_frames
    ]
    return float(np.mean(per_frame))


def _solver_agreement(
    decisions: Sequence[SolverDecision], comparison: dict | None
) -> tuple[float | None, str | None]:
    """(agreement, kind) where kind is "path" or "verdict".

    None unless at least two backends succeeded — with one result there is
    nothing to agree with, and returning a number anyway would be fabricated
    certainty.

    "path" comes from `metrics.compare_solvers` (Sim(3)-aligned trajectories).
    "verdict" is the fallback computable from the audit log alone: agreement on
    reported confidence, and — among triangulating backends only, since a
    screen-space solve's registration and residual mean something else — on
    registration ratio and reprojection error. The worst of those decides,
    because one quantity agreeing does not excuse another disagreeing.
    """
    succeeded = [d for d in decisions if d.succeeded]
    if (
        comparison is not None
        and comparison.get("agreement_score") is not None
        and int(comparison.get("succeeded_count", 0)) >= 2
    ):
        return _clamp01(float(comparison["agreement_score"])), "path"
    if len(succeeded) < 2:
        return None, None

    def spread_agreement(values: list[float], tolerance: float) -> float:
        centre = float(np.median(values))
        deviation = float(np.mean(np.abs(np.array(values) - centre)))
        return _clamp01(1.0 - deviation / tolerance)

    agreements = [spread_agreement([d.confidence for d in succeeded], SOLVER_VERDICT_TOLERANCE)]

    geometric = [d for d in succeeded if d.solver in GEOMETRIC_SOURCES]
    ratios = [d.registered_frames / d.total_frames for d in geometric if d.total_frames > 0]
    if len(ratios) >= 2:
        agreements.append(spread_agreement(ratios, SOLVER_VERDICT_TOLERANCE))

    errors = [
        d.mean_reprojection_error for d in geometric
        if d.mean_reprojection_error is not None and np.isfinite(d.mean_reprojection_error)
    ]
    if len(errors) >= 2:
        centre = float(np.median(errors))
        if centre > 1e-9:
            relative = float(np.mean(np.abs(np.array(errors) - centre))) / centre
            agreements.append(_clamp01(1.0 - relative / SOLVER_REPROJECTION_TOLERANCE))

    return float(min(agreements)), "verdict"


def _radial_significance(signature: MotionSignature, lens: Sequence[LensFrame] = ()) -> float:
    """Strength of the zoom the lens curve claims, 0-1.

    Measured from the lens curve's focal range, not from the image-space
    `net_scale_change`. That statistic comes from a similarity fit, which reads
    perspective under a pan as a steady scale change (1.0196 per frame on a
    synthetic pure pan — the same as a genuine 1.9% zoom). The lens curve is built
    from the intrinsics-normalised homographies, which separate the two. Only a
    zoom the system actually claims can be confounded with a dolly.
    """
    focals = [f.focal_normalized for f in lens if np.isfinite(f.focal_normalized) and f.focal_normalized > 0]
    if len(focals) < 2:
        return 0.0
    return _ramp(max(focals) / min(focals) - 1.0, SCALE_CHANGE_NEGLIGIBLE, SCALE_CHANGE_STRONG)


def _lens_zoom_ratio(lens: Sequence[LensFrame]) -> float:
    focals = [f.focal_normalized for f in lens if np.isfinite(f.focal_normalized) and f.focal_normalized > 0]
    return max(focals) / min(focals) if len(focals) >= 2 else 1.0


# --------------------------------------------------------------------------
# The three sub-confidences, and the screen-space match
# --------------------------------------------------------------------------


def _rotation_confidence(
    *,
    inlier_term: float,
    texture_term: float,
    sharpness_term: float,
    temporal_consistency: float,
    registered_ratio: float,
    geometric: bool,
    focal_stability: float,
) -> float:
    """Rotation is the best-conditioned quantity in monocular video: any set of
    coherently moving tracks constrains it, with or without depth.

    The one caveat is magnitude. Without a geometric solve, yaw and pitch are
    recovered by dividing image motion by a focal length, so the rotation's
    *direction* is measured but its *size* is only as trustworthy as the lens.
    That is why a non-geometric solve is scaled by focal stability rather than
    being treated as equally good.
    """
    quality = _weighted_mean(
        [
            (0.35, inlier_term),
            (0.20, texture_term),
            (0.15, sharpness_term),
            (0.30, temporal_consistency),
        ]
    )
    if geometric:
        quality = 0.75 * quality + 0.25 * _ramp(registered_ratio, REGISTERED_POOR, REGISTERED_GOOD)
    else:
        quality *= 0.70 + 0.30 * focal_stability
    return _clamp01(quality)


def _translation_confidence(
    *,
    observable: bool,
    parallax: float,
    registered_ratio: float,
    reprojection_term: float | None,
    ba_term: float | None,
    track_term: float,
    temporal_consistency: float,
    perceptual: bool,
) -> float:
    """Translation needs a baseline, and a baseline needs parallax.

    Every other piece of evidence is conditional on that: a bundle with 0.3 px
    reprojection error over a pure pan is a beautifully converged fit to a
    structure that does not determine translation at all.
    """
    raw = _weighted_mean(
        [
            (0.35, _ramp(parallax, PARALLAX_OBSERVABLE_MIN, PARALLAX_STRONG)),
            (0.20, _ramp(registered_ratio, REGISTERED_POOR, REGISTERED_GOOD)),
            (0.15, reprojection_term),
            (0.10, ba_term),
            (0.10, track_term),
            (0.10, temporal_consistency),
        ]
    )
    if not observable:
        return min(raw, UNOBSERVABLE_TRANSLATION_CEILING)
    if perceptual:
        return min(raw, PERCEPTUAL_TRANSLATION_CEILING)
    return raw


def _zoom_confidence(
    *,
    lens: Sequence[LensFrame],
    focal_stability: float | None,
    focal_check: _FocalCheck | None,
    focal_observable: bool,
    inlier_term: float,
    sharpness_term: float,
    radial_significance: float,
    parallax: float,
) -> float:
    """Confidence in the recovered FOV / focal curve.

    Zero with no lens curve at all: tracking quality alone says nothing about a
    lens nobody estimated.

    The hard part is not measuring the radial flow, it is attributing it. A
    zoom-in and a dolly-in toward a distant subject produce the same expanding
    image, and only parallax separates them. So a strong radial signal with weak
    depth evidence reduces this number even when the curve fits the footage
    perfectly — the framing is right, the cause is not established.
    """
    if not lens:
        return 0.0
    lens_confidence = float(np.mean([_clamp01(f.confidence) for f in lens]))
    base = _weighted_mean(
        [
            (0.35, lens_confidence),
            (0.20, focal_stability),
            (0.20, focal_check.agreement if focal_check else None),
            (0.15, inlier_term),
            (0.10, sharpness_term),
        ]
    )
    separability = _ramp(parallax, PARALLAX_OBSERVABLE_MIN, ZOOM_SEPARABLE_PARALLAX)
    confound = radial_significance * (1.0 - separability)
    value = _clamp01(base * (1.0 - ZOOM_CONFOUND_PENALTY * confound))
    if not focal_observable:
        value = min(value, FOCAL_UNOBSERVED_CEILING)
    return value


def _screen_match_confidence(
    *,
    geometry: GeometryResult,
    residual_px: float | None,
    inlier_term: float,
    measurement_continuity: float | None,
) -> float:
    """Confidence that a perceptual camera reproduces the reference *on screen*.

    Deliberately a separate number from anything physical: this is the claim
    Perceptual Match actually makes, and it may legitimately be high.
    """
    residual_term = (
        _falling_ramp(residual_px, REPROJECTION_EXCELLENT_PX, REPROJECTION_POOR_PX)
        if residual_px is not None
        else None
    )
    return _weighted_mean(
        [
            (0.35, residual_term),
            (0.25, inlier_term),
            (0.20, measurement_continuity),
            (0.20, geometry.confidence if geometry.succeeded else None),
        ]
    )


# --------------------------------------------------------------------------
# Evidence bundle
# --------------------------------------------------------------------------


@dataclass
class _Evidence:
    """Everything the verdict and its explanation are derived from. Built once,
    so the score and the reasons cannot describe different numbers."""

    geometry: GeometryResult
    signature: MotionSignature
    motion_frames: Sequence[MotionFrame]
    lens: Sequence[LensFrame]
    decisions: Sequence[SolverDecision]

    perceptual: bool
    geometric: bool
    registered_ratio: float
    registered: int
    total: int
    track_count: int
    track_kind: str
    inlier_ratio: float
    parallax: float
    texture: float
    sharpness: float
    mean_reprojection: float | None
    median_reprojection: float | None
    ba_residual: float | None
    agreement: float | None
    agreement_kind: str | None
    focal_observable: bool
    focal_stability: float | None
    focal_check: _FocalCheck | None
    temporal_consistency: float
    unsupported_rotation: float | None
    unsupported_position: float | None
    measurement_continuity: float | None
    radial: float
    verdict: TranslationVerdict

    rotation_confidence: float = 0.0
    translation_confidence: float = 0.0
    zoom_confidence: float = 0.0
    screen_match: float | None = None
    score: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.LOW

    @property
    def translation_observable(self) -> bool:
        return self.verdict is TranslationVerdict.OBSERVABLE

    @property
    def solver_name(self) -> str:
        return self.geometry.source.value.upper() if self.geometry.source in (
            SolverSource.COLMAP, SolverSource.VGGT
        ) else self.geometry.source.value.replace("_", " ")


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def _level(ev: _Evidence) -> ConfidenceLevel:
    """Band the score, then apply the gates that a score alone cannot enforce."""
    band = _band(ev.score)
    if band is not ConfidenceLevel.HIGH:
        return band
    gates = (
        ev.translation_observable
        and ev.geometric
        and not ev.perceptual
        and ev.registered_ratio >= HIGH_REGISTERED_RATIO_MIN
        and (ev.inlier_ratio >= HIGH_INLIER_RATIO_MIN or ev.parallax >= PARALLAX_OBSERVABLE_MIN)
        and ev.parallax >= HIGH_PARALLAX_MIN
        and ev.temporal_consistency >= HIGH_TEMPORAL_CONSISTENCY_MIN
        and ev.track_count >= HIGH_TRACK_COUNT_MIN
        and (ev.mean_reprojection is None or ev.mean_reprojection <= HIGH_REPROJECTION_MAX_PX)
        and (ev.agreement is None or ev.agreement >= HIGH_SOLVER_AGREEMENT_MIN)
    )
    return ConfidenceLevel.HIGH if gates else ConfidenceLevel.MEDIUM


def assess_confidence(
    geometry: GeometryResult,
    signature: MotionSignature,
    motion_frames: list[MotionFrame],
    poses: list[CameraPose],
    lens: list[LensFrame],
    solver_decisions: list[SolverDecision],
    parallax_score: float,
    texture: float,
    blur: float,
    pipeline_mode_used: PipelineMode,
    *,
    solver_comparison: dict | None = None,
) -> ConfidenceReport:
    """Judge one shot's trajectory from the evidence behind it (I7).

    `blur` follows `MotionSignature.blur_score`: 1 = sharp. `solver_comparison`
    is the optional output of `metrics.compare_solvers`; with it,
    `solver_agreement` measures path agreement rather than verdict agreement.

    Never raises and never refuses to answer: an empty, failed, evidence-free
    solve is a legitimate input and produces a LOW verdict that explains itself.
    Should the assessment itself fail on malformed input, the result is a LOW
    report naming the failure, not an exception (invariant I12).
    """
    try:
        return _assess(
            geometry, signature, motion_frames, poses, lens, solver_decisions,
            parallax_score, texture, blur, pipeline_mode_used, solver_comparison,
        )
    except Exception as exc:  # noqa: BLE001 - I12: validation failure degrades, never crashes
        log.exception("confidence assessment failed")
        return ConfidenceReport(
            level=ConfidenceLevel.LOW,
            score=0.0,
            headline=(
                "LOW CONFIDENCE - The evidence for this shot could not be assessed, so "
                "no confidence is claimed for the trajectory."
            ),
            reasons=[
                f"The confidence assessment failed ({type(exc).__name__}: {exc}). The "
                f"trajectory is reported without any verified evidence behind it."
            ],
            translation_observable=False,
        )


def _assess(
    geometry: GeometryResult,
    signature: MotionSignature,
    motion_frames: list[MotionFrame],
    poses: list[CameraPose],
    lens: list[LensFrame],
    solver_decisions: list[SolverDecision],
    parallax_score: float,
    texture: float,
    blur: float,
    pipeline_mode_used: PipelineMode,
    solver_comparison: dict | None,
) -> ConfidenceReport:
    perceptual = (
        pipeline_mode_used is PipelineMode.PERCEPTUAL_MATCH
        or geometry.source is SolverSource.PERCEPTUAL
    )
    geometric = bool(geometry.succeeded and geometry.source in GEOMETRIC_SOURCES and not perceptual)

    # --- evidence ---------------------------------------------------------
    registered_ratio, registered, total = _registration(geometry, geometric, poses)
    track_count, track_kind = _persistent_tracks(geometry, geometric, motion_frames)
    inlier_ratio = _mean_inlier_ratio(motion_frames, signature)
    parallax = _explicit_or_signature(parallax_score, signature.parallax_score)
    texture_score = _explicit_or_signature(texture, signature.texture_score)
    sharpness = _explicit_or_signature(blur, signature.blur_score)

    # A focal the solve could not constrain is a prior. Only a successful solve
    # can make that claim; a failed result's default says nothing either way.
    focal_observable = not (geometry.succeeded and not geometry.focal_observable)
    measured_stability = _focal_stability(lens)
    focal_stability = measured_stability
    if not focal_observable:
        focal_stability = min(measured_stability or 0.0, FOCAL_UNOBSERVED_CEILING)
    focal_check = _focal_agreement(geometry, geometric, lens)

    pose_term, unsupported_rotation, unsupported_position = _pose_continuity(poses, motion_frames)
    measurement_continuity = _measurement_continuity(motion_frames)
    temporal_consistency = _weighted_mean([(0.60, pose_term), (0.40, measurement_continuity)])
    agreement, agreement_kind = _solver_agreement(solver_decisions, solver_comparison)

    mean_reprojection = (
        _finite_or_none(geometry.mean_reprojection_error) if geometry.succeeded else None
    )
    median_reprojection = (
        _finite_or_none(geometry.median_reprojection_error) if geometry.succeeded else None
    )
    ba_residual = (
        _finite_or_none(geometry.bundle_adjustment_residual) if geometry.succeeded else None
    )

    # --- observability ----------------------------------------------------
    # The footage and the solve must both support a baseline. A solver that
    # reports one on a homography-only shot is reporting noise, and a solve that
    # recovered none must not be upgraded by a parallax score it never used.
    if parallax < PARALLAX_OBSERVABLE_MIN:
        verdict = TranslationVerdict.NO_PARALLAX
    elif not geometry.succeeded:
        verdict = TranslationVerdict.NO_RECONSTRUCTION
    elif not geometry.translation_observable:
        verdict = TranslationVerdict.SOLVER_REPORTED
    else:
        verdict = TranslationVerdict.OBSERVABLE

    ev = _Evidence(
        geometry=geometry, signature=signature, motion_frames=motion_frames, lens=lens,
        decisions=solver_decisions, perceptual=perceptual, geometric=geometric,
        registered_ratio=registered_ratio, registered=registered, total=total,
        track_count=track_count, track_kind=track_kind, inlier_ratio=inlier_ratio,
        parallax=parallax, texture=texture_score, sharpness=sharpness,
        mean_reprojection=mean_reprojection, median_reprojection=median_reprojection,
        ba_residual=ba_residual, agreement=agreement, agreement_kind=agreement_kind,
        focal_observable=focal_observable, focal_stability=focal_stability,
        focal_check=focal_check, temporal_consistency=temporal_consistency,
        unsupported_rotation=unsupported_rotation, unsupported_position=unsupported_position,
        measurement_continuity=measurement_continuity,
        radial=_radial_significance(signature, lens), verdict=verdict,
    )

    inlier_term = _ramp(inlier_ratio, INLIER_POOR, INLIER_GOOD)
    if ev.geometric and ev.translation_observable:
        # With real depth, one 2D model cannot explain all the flow — that is
        # what parallax IS. Low 2D agreement there is the expected signature of
        # depth, not evidence of moving content, so it is discounted by the
        # measured parallax. (A synthetic orbit with 0.05 deg rotation error was
        # held at MEDIUM because a 2D model explained 49% of its tracks.)
        inlier_term = max(inlier_term, _ramp(parallax, PARALLAX_OBSERVABLE_MIN, PARALLAX_STRONG))
    texture_term = _ramp(texture_score, TEXTURE_POOR, TEXTURE_GOOD)
    sharpness_term = _ramp(sharpness, SHARPNESS_POOR, SHARPNESS_GOOD)
    track_term = _ramp(float(track_count), TRACKS_POOR, TRACKS_GOOD)
    registered_term = _ramp(registered_ratio, REGISTERED_POOR, REGISTERED_GOOD)
    # A perceptual backend's "reprojection error" is a screen-space signature
    # residual. It is evidence for the match, never for the physical path.
    reprojection_term = (
        _falling_ramp(mean_reprojection, REPROJECTION_EXCELLENT_PX, REPROJECTION_POOR_PX)
        if mean_reprojection is not None and geometric
        else None
    )
    ba_term = (
        _falling_ramp(ba_residual, BA_RESIDUAL_EXCELLENT, BA_RESIDUAL_POOR)
        if ba_residual is not None and geometric
        else None
    )

    # --- sub-confidences --------------------------------------------------
    ev.rotation_confidence = _rotation_confidence(
        inlier_term=inlier_term, texture_term=texture_term, sharpness_term=sharpness_term,
        temporal_consistency=temporal_consistency, registered_ratio=registered_ratio,
        geometric=geometric,
        focal_stability=focal_stability if focal_stability is not None else 0.0,
    )
    ev.translation_confidence = _translation_confidence(
        observable=ev.translation_observable, parallax=parallax,
        registered_ratio=registered_ratio, reprojection_term=reprojection_term,
        ba_term=ba_term, track_term=track_term,
        temporal_consistency=temporal_consistency, perceptual=perceptual,
    )
    ev.zoom_confidence = _zoom_confidence(
        lens=lens, focal_stability=focal_stability, focal_check=focal_check,
        focal_observable=focal_observable, inlier_term=inlier_term,
        sharpness_term=sharpness_term, radial_significance=ev.radial, parallax=parallax,
    )
    if perceptual:
        ev.screen_match = _screen_match_confidence(
            geometry=geometry, residual_px=mean_reprojection, inlier_term=inlier_term,
            measurement_continuity=measurement_continuity,
        )

    # --- aggregate --------------------------------------------------------
    agreement_weight = SOLVER_AGREEMENT_WEIGHT * (
        1.0 if agreement_kind == "path" else VERDICT_AGREEMENT_WEIGHT_FACTOR
    )
    score = _weighted_mean(
        [
            (0.18, registered_term),
            (0.12, inlier_term),
            (0.10, reprojection_term),
            (0.06, ba_term),
            (agreement_weight, agreement),
            (0.08, temporal_consistency),
            (0.06, focal_stability),
            (0.06, 0.5 * (texture_term + sharpness_term)),
            (0.13, ev.rotation_confidence),
            (0.10, ev.translation_confidence),
            (0.05, ev.zoom_confidence),
        ]
    )
    if not ev.translation_observable:
        score = min(score, UNOBSERVABLE_SCORE_CAP)
    if perceptual:
        score = min(score, PERCEPTUAL_PHYSICAL_SCORE_CAP)
    if not geometry.succeeded:
        score = min(score, NO_GEOMETRY_SCORE_CAP)
    ev.score = _clamp01(score)
    ev.level = _level(ev)

    reasons = _build_reasons(ev)
    headline = _build_headline(ev)

    log.info(
        "confidence: %s score=%.2f (rot=%.2f trans=%.2f zoom=%.2f) registered=%d/%d "
        "parallax=%.2f translation=%s focal_observable=%s mode=%s",
        ev.level.value, ev.score, ev.rotation_confidence, ev.translation_confidence,
        ev.zoom_confidence, registered, total, parallax, verdict.value,
        focal_observable, pipeline_mode_used.value,
    )

    return ConfidenceReport(
        level=ev.level,
        score=ev.score,
        headline=headline,
        reasons=reasons,
        registered_frame_ratio=registered_ratio,
        persistent_track_count=track_count,
        mean_inlier_ratio=inlier_ratio,
        mean_reprojection_error=mean_reprojection,
        median_reprojection_error=median_reprojection,
        baseline_parallax_score=parallax,
        bundle_adjustment_residual=ba_residual,
        solver_agreement=agreement,
        focal_stability=focal_stability if focal_stability is not None else 0.0,
        temporal_consistency=temporal_consistency,
        translation_observable=ev.translation_observable,
        translation_confidence=ev.translation_confidence,
        rotation_confidence=ev.rotation_confidence,
        zoom_confidence=ev.zoom_confidence,
    )


# --------------------------------------------------------------------------
# Plain-English explanation
# --------------------------------------------------------------------------


def _degeneracy_sentence(signature: MotionSignature, radial: float) -> str:
    """Why the footage carries no baseline, from the shot's own measurements."""
    if signature.mean_flow_magnitude < STATIC_FLOW_PX and signature.cumulative_path_pixels < (
        STATIC_FLOW_PX * max(signature.frame_count, 1)
    ):
        return "The camera is effectively static, so there is no motion to measure depth from"
    if signature.homography_dominance > 0:
        scene = (
            f"{_percent(signature.homography_dominance)} of the shot's frame-to-frame motion "
            f"is explained as if the entire scene were distant or flat"
        )
    else:
        scene = "No depth-dependent flow was measured anywhere in the shot"
    if signature.rotation_dominance >= PURE_ROTATION_DOMINANCE:
        motion = "the shot contains almost pure rotation"
    elif radial > 0.5 and signature.rotation_dominance < MOSTLY_ROTATION_DOMINANCE:
        motion = "the image motion is dominated by zoom-like scaling"
    elif signature.rotation_dominance >= MOSTLY_ROTATION_DOMINANCE:
        motion = f"the motion is {_percent(signature.rotation_dominance)} rotational"
    else:
        motion = "any camera travel is too small relative to the scene depth to measure"
    return f"{scene}, and {motion}"


def _prior_fov(ev: _Evidence) -> float | None:
    fov = _geometry_fov(ev.geometry)
    if fov is not None:
        return fov
    if ev.lens:
        return float(np.median([f.fov_horizontal for f in ev.lens]))
    return None


def _build_reasons(ev: _Evidence) -> list[str]:
    """Explain the verdict in language a director can act on.

    Never returns an empty list: a report with no reasons is indistinguishable
    from a report nobody checked.
    """
    geometry = ev.geometry
    ranked: list[tuple[int, str]] = []

    # --- translation: the decisive fact for any physical verdict -----------
    if ev.verdict is TranslationVerdict.NO_PARALLAX:
        ranked.append((
            _DECISIVE,
            f"Translation cannot be recovered from this shot. "
            f"{_degeneracy_sentence(ev.signature, ev.radial)}. The parallax score is "
            f"{ev.parallax:.2f}, below the {PARALLAX_OBSERVABLE_MIN:.2f} needed to measure a "
            f"camera baseline, so any camera travel is indistinguishable from rotation or zoom.",
        ))
    elif ev.verdict is TranslationVerdict.NO_RECONSTRUCTION:
        ranked.append((
            _DECISIVE,
            f"The footage shows depth parallax ({ev.parallax:.2f}), but no geometric "
            f"reconstruction succeeded, so the camera's translation was never measured.",
        ))
    elif ev.verdict is TranslationVerdict.SOLVER_REPORTED:
        ranked.append((
            _DECISIVE,
            f"The footage shows depth parallax ({ev.parallax:.2f}), but the {ev.solver_name} "
            f"solve did not recover a camera baseline from it, so translation is not "
            f"observable in this result.",
        ))
    else:
        ranked.append((
            _PRIMARY,
            f"Depth-dependent flow is present (parallax score {ev.parallax:.2f}), so the "
            f"camera genuinely translated and its path shape is measurable — in normalized "
            f"units, not metres, unless a scale is calibrated.",
        ))
    if (not ev.translation_observable and geometry.succeeded and not ev.perceptual
            and geometry.source is SolverSource.OPENCV):
        ranked.append((
            _DECISIVE,
            f"Rotation was measured directly from long-baseline homographies between "
            f"keyframes ({geometry.message.split(';')[0]}), which is exact for a camera "
            f"that rotates without moving through depth. Camera position is held fixed: "
            f"no translation was measured, so none is shown.",
        ))
    elif not ev.translation_observable and geometry.succeeded and not ev.perceptual:
        ranked.append((
            _DECISIVE,
            f"The {ev.solver_name} solver still returned camera positions. Without a "
            f"measurable baseline those positions are fitted noise rather than measured "
            f"travel: do not read the path shape as the camera's movement.",
        ))

    # --- perceptual: physical path vs screen-space match -------------------
    if ev.perceptual:
        ranked.append((
            _MODE,
            f"Physical trajectory: Perceptual Match does not measure the physical camera "
            f"path, so confidence in the path itself is at most MEDIUM and translation "
            f"confidence is {ev.translation_confidence:.2f}. Read the positions as a "
            f"screen-space equivalent, not as a measured dolly or truck.",
        ))
        if ev.screen_match is not None:
            residual = (
                f" with a {ev.mean_reprojection:.2f} px signature residual"
                if ev.mean_reprojection is not None
                else ""
            )
            ranked.append((
                _MODE,
                f"Screen-space match: {_band(ev.screen_match).value.upper()} "
                f"({ev.screen_match:.2f}). The recovered camera reproduces the reference's "
                f"on-screen motion{residual}. This is confidence in the match — which is "
                f"what a downstream generative model consumes — not in the physical camera "
                f"path.",
            ))

    # --- registration ------------------------------------------------------
    if ev.geometric:
        ranked.append((
            _PRIMARY,
            f"{_percent(ev.registered_ratio)} of the frames submitted to the {ev.solver_name} "
            f"solve registered ({ev.registered} of {ev.total}).",
        ))
    elif geometry.succeeded:
        ranked.append((
            _SECONDARY,
            f"No frames were geometrically registered: the {ev.solver_name} backend fits "
            f"motion without triangulating 3D structure, so registered-frame ratio is 0 by "
            f"construction.",
        ))
    else:
        message = geometry.message.strip() or "no reason recorded"
        ranked.append((
            _DECISIVE,
            f"No geometric reconstruction succeeded, so there is no measured 3D structure "
            f"behind this trajectory ({message}).",
        ))

    # --- lens --------------------------------------------------------------
    if not ev.focal_observable:
        sensitivity = geometry.focal_sensitivity
        evidence = (
            f"pinning the focal length {FOCAL_PROBE_FRACTION:.0%} either way changed the "
            f"reprojection error by only {sensitivity:.1%}"
            if sensitivity is not None and np.isfinite(sensitivity)
            else "the solve found the focal length unconstrained by the footage"
        )
        fov = _prior_fov(ev)
        prior = (
            f"the {fov:.1f} deg horizontal FOV used is the prior the poses were solved under"
            if fov is not None
            else "the FOV used is the prior the poses were solved under"
        )
        ranked.append((
            _DECISIVE,
            f"The field of view could not be measured from this shot: {evidence}, so "
            f"{prior}, not a measurement. Zoom confidence and focal stability are LOW for "
            f"that reason. If you know the lens, set the FOV override.",
        ))
    elif not ev.lens:
        ranked.append((
            _PRIMARY,
            "No per-frame lens estimate was available, so zoom confidence is 0 and focal "
            "stability could not be measured.",
        ))
    else:
        if ev.focal_stability is None:
            ranked.append((
                _SECONDARY,
                "Too few lens samples to tell a zoom from estimation noise, so focal "
                "stability is not measured.",
            ))
        elif ev.focal_stability < 0.5:
            ranked.append((
                _PRIMARY,
                f"The focal-length estimate rattles frame to frame (stability "
                f"{ev.focal_stability:.2f}), which propagates straight into rotation "
                f"magnitude and FOV.",
            ))
        if ev.focal_check is not None:
            if ev.focal_check.agreement < 0.5:
                ranked.append((
                    _PRIMARY,
                    f"The lens curve ({ev.focal_check.lens_fov:.1f} deg) disagrees with the "
                    f"focal length the {ev.solver_name} solve measured "
                    f"({ev.focal_check.geometry_fov:.1f} deg horizontal) by more than the "
                    f"{FOCAL_PROBE_FRACTION:.0%} the solve can resolve.",
                ))
            else:
                ranked.append((
                    _SECONDARY,
                    f"The lens curve ({ev.focal_check.lens_fov:.1f} deg) agrees with the "
                    f"focal length measured by the {ev.solver_name} solve "
                    f"({ev.focal_check.geometry_fov:.1f} deg horizontal).",
                ))
        elif ev.geometric and geometry.focal_pixels is not None and not geometry.focal_image_width:
            ranked.append((
                _SECONDARY,
                "The solve reported a focal length in pixels without the image width it "
                "refers to, so it could not be checked against the lens curve.",
            ))
        if ev.geometric and geometry.focal_sensitivity is None:
            ranked.append((
                _SECONDARY,
                "The solve did not probe whether this shot constrains the focal length, so "
                "the FOV is not verified as a measurement.",
            ))

    if ev.radial > 0.25 and ev.parallax < ZOOM_SEPARABLE_PARALLAX:
        ranked.append((
            _PRIMARY,
            f"The lens curve changes focal length by {abs(_lens_zoom_ratio(ev.lens) - 1.0):.1%} "
            f"over the shot with too little depth evidence to separate a zoom from a dolly. The "
            f"recovered FOV curve reproduces the framing; its physical cause is not "
            f"established (zoom confidence {ev.zoom_confidence:.2f}).",
        ))

    # --- rotation ----------------------------------------------------------
    if ev.rotation_confidence >= HIGH_SCORE_MIN and not ev.translation_observable:
        ranked.append((
            _PRIMARY,
            f"Rotation is well determined (rotation confidence {ev.rotation_confidence:.2f}) "
            f"even though translation is not: orientation is constrained by any coherently "
            f"moving tracks, with or without depth.",
        ))
    else:
        ranked.append((
            _SECONDARY,
            f"Rotation confidence {ev.rotation_confidence:.2f}, translation confidence "
            f"{ev.translation_confidence:.2f}, zoom confidence {ev.zoom_confidence:.2f}. "
            f"Timing comes from container timestamps and is unaffected by any of these.",
        ))

    # --- solve quality -----------------------------------------------------
    if ev.mean_reprojection is not None:
        median = (
            f" (median {ev.median_reprojection:.2f} px)"
            if ev.median_reprojection is not None
            else ""
        )
        width = (
            f" on {geometry.focal_image_width} px-wide images" if geometry.focal_image_width else ""
        )
        if ev.geometric:
            if ev.mean_reprojection <= REPROJECTION_EXCELLENT_PX:
                verdict = ": sub-pixel, the bundle converged."
            elif ev.mean_reprojection <= HIGH_REPROJECTION_MAX_PX:
                verdict = "."
            else:
                verdict = (
                    ": the tracks and the recovered structure disagree by more than "
                    "feature-localisation noise."
                )
            ranked.append((
                _PRIMARY,
                f"Mean reprojection error {ev.mean_reprojection:.2f} px{median}{width}{verdict}",
            ))
        else:
            ranked.append((
                _SECONDARY,
                f"The {ev.mean_reprojection:.2f} px residual reported by the {ev.solver_name} "
                f"backend is a screen-space signature residual, not a 3D reprojection error.",
            ))
    elif ev.geometric:
        ranked.append((
            _SECONDARY,
            f"The {ev.solver_name} backend reported no reprojection error, so that evidence "
            f"is missing from this score rather than assumed good.",
        ))

    if ev.ba_residual is not None:
        ranked.append((_SECONDARY, f"Bundle-adjustment residual {ev.ba_residual:.2f} px."))

    if ev.track_kind == "3d_points":
        tracks = f"{ev.track_count} triangulated 3D points support the solve"
    elif ev.track_kind == "dense_tracks":
        tracks = (
            f"A median of {ev.track_count} background tracks survived each frame transition "
            f"(no triangulated structure)"
        )
    else:
        tracks = "No tracks were available"
    ranked.append((
        _PRIMARY if ev.geometric and ev.track_count < HIGH_TRACK_COUNT_MIN else _SECONDARY,
        f"{tracks}; the dominant 2D motion model explained {_percent(ev.inlier_ratio)} of "
        f"tracked features on average"
        + (" — expected when the scene has depth, since parallax is exactly the motion a "
           "single 2D model cannot explain." if ev.geometric and ev.parallax >= PARALLAX_OBSERVABLE_MIN
           else "."),
    ))

    succeeded = [d for d in ev.decisions if d.succeeded]
    if ev.agreement is None:
        ranked.append((
            _SECONDARY,
            f"Only {len(succeeded)} backend produced a usable result, so cross-solver "
            f"agreement could not be measured and is reported as unknown rather than as "
            f"agreement."
            if len(succeeded) == 1
            else "No backend produced a usable result that could be cross-checked, so "
            "cross-solver agreement could not be measured.",
        ))
    elif ev.agreement_kind == "path":
        ranked.append((
            _PRIMARY if ev.agreement < HIGH_SOLVER_AGREEMENT_MIN else _SECONDARY,
            f"Independent backends agree on the camera path to {_percent(ev.agreement)} after "
            f"Sim(3) alignment.",
        ))
    else:
        ranked.append((
            _PRIMARY if ev.agreement < HIGH_SOLVER_AGREEMENT_MIN else _SECONDARY,
            f"{len(succeeded)} backends succeeded and agree on solve quality to "
            f"{_percent(ev.agreement)}. This compares their verdicts, not their paths.",
        ))

    # --- image quality -----------------------------------------------------
    if ev.texture < TEXTURE_POOR:
        ranked.append((
            _PRIMARY,
            f"The frames are poorly textured (texture score {ev.texture:.2f}): there are few "
            f"features to track, which limits every estimate here.",
        ))
    if ev.sharpness < SHARPNESS_POOR:
        ranked.append((
            _PRIMARY,
            f"Heavy motion blur (sharpness {ev.sharpness:.2f}) smears the features the solve "
            f"depends on.",
        ))

    # --- temporal ----------------------------------------------------------
    unsupported = [v for v in (ev.unsupported_rotation, ev.unsupported_position) if v is not None]
    if ev.temporal_consistency < HIGH_TEMPORAL_CONSISTENCY_MIN:
        detail = (
            f"{_percent(max(unsupported))} of steps jump with no matching change in the "
            f"measured image motion"
            if unsupported and max(unsupported) > 0
            else "the dense 2D measurement explained too little of the shot"
        )
        ranked.append((
            _PRIMARY,
            f"Temporal consistency is {ev.temporal_consistency:.2f}: {detail}.",
        ))
    elif unsupported:
        ranked.append((
            _SECONDARY,
            f"No unexplained jumps in the recovered motion (temporal consistency "
            f"{ev.temporal_consistency:.2f}); high-frequency content that matches the "
            f"measured image motion is treated as real and preserved.",
        ))

    reasons = [text for _, text in sorted(ranked, key=lambda item: item[0])]
    if not reasons:
        reasons = [
            "No evidence was available for this shot, so no confidence can be claimed for "
            "the result."
        ]
    return reasons


def _build_headline(ev: _Evidence) -> str:
    """One line, leading with the verdict and the fact that decided it."""
    prefix = f"{ev.level.value.upper()} CONFIDENCE"

    if ev.verdict is TranslationVerdict.NO_PARALLAX:
        if ev.perceptual:
            tail = " Perceptual Match has therefore been used."
        elif not ev.geometry.succeeded:
            tail = (
                " No geometric reconstruction succeeded either, so this trajectory rests on "
                "the 2D motion signature alone."
            )
        elif ev.geometry.source is SolverSource.OPENCV:
            tail = (
                " Rotation was measured from long-baseline keyframe geometry, and the "
                "camera position is held fixed rather than invented."
            )
        else:
            tail = (
                f" The positions from the {ev.solver_name} solve should not be read as "
                f"measured camera travel."
            )
        return (
            f"{prefix} - {_degeneracy_sentence(ev.signature, ev.radial)}. Translation "
            f"cannot be reliably reconstructed.{tail}"
        )

    if ev.verdict is TranslationVerdict.NO_RECONSTRUCTION:
        return (
            f"{prefix} - No geometric backend produced a usable reconstruction, so this "
            f"trajectory rests on the 2D motion signature alone. Rotation and timing are "
            f"recovered; the path shape is not measured."
        )

    if ev.verdict is TranslationVerdict.SOLVER_REPORTED:
        if ev.perceptual:
            return (
                f"{prefix} - Perceptual Match did not solve translation for this shot, so "
                f"only rotation, zoom and timing are recovered. Translation cannot be "
                f"reliably reconstructed."
            )
        return (
            f"{prefix} - The {ev.solver_name} solve registered the shot, but its cameras "
            f"barely move relative to the scene, so translation cannot be reliably "
            f"reconstructed. The positions should not be read as measured camera travel."
        )

    if ev.perceptual:
        match = (
            f" ({_band(ev.screen_match).value} screen-space match)"
            if ev.screen_match is not None
            else ""
        )
        return (
            f"{prefix} - Perceptual Match reproduces the reference's on-screen motion{match} "
            f"but does not measure the physical camera path. Treat the trajectory as a "
            f"screen-space equivalent of the move."
        )

    if ev.level is ConfidenceLevel.HIGH:
        error = (
            f" at {ev.mean_reprojection:.2f} px reprojection error"
            if ev.mean_reprojection is not None
            else ""
        )
        if not ev.focal_observable:
            lens = (
                " The camera path is well determined; the field of view could not be "
                "measured from this shot and is an assumed prior."
            )
        elif ev.zoom_confidence >= HIGH_SCORE_MIN:
            lens = " Camera translation, rotation and lens are all well determined."
        else:
            lens = " Camera translation and rotation are well determined."
        return (
            f"{prefix} - {_percent(ev.registered_ratio)} of frames registered{error}, with "
            f"clear depth parallax ({ev.parallax:.2f}).{lens}"
        )

    strongest = "Rotation" if ev.rotation_confidence >= ev.translation_confidence else "Translation"
    return (
        f"{prefix} - The shot reconstructed, but {_primary_limitation(ev)} {strongest} and "
        f"timing are the most trustworthy parts of this result."
    )


def _primary_limitation(ev: _Evidence) -> str:
    """The single worst piece of evidence, phrased as a clause."""
    if ev.registered_ratio < HIGH_REGISTERED_RATIO_MIN:
        return (
            f"only {_percent(ev.registered_ratio)} of frames registered in the geometric "
            f"solve."
        )
    if ev.parallax < HIGH_PARALLAX_MIN:
        return (
            f"the parallax baseline is thin ({ev.parallax:.2f}), so how far the camera "
            f"travelled relative to the scene is weakly constrained."
        )
    if ev.mean_reprojection is not None and ev.mean_reprojection > HIGH_REPROJECTION_MAX_PX:
        return (
            f"the bundle settled at {ev.mean_reprojection:.2f} px reprojection error, above "
            f"the {HIGH_REPROJECTION_MAX_PX:.1f} px this system trusts."
        )
    if ev.agreement is not None and ev.agreement < HIGH_SOLVER_AGREEMENT_MIN:
        return (
            f"independent backends disagree (agreement {_percent(ev.agreement)}), so the "
            f"selected result is not corroborated."
        )
    if ev.temporal_consistency < HIGH_TEMPORAL_CONSISTENCY_MIN:
        return (
            f"the recovered motion contains jumps the image motion does not support "
            f"(temporal consistency {ev.temporal_consistency:.2f})."
        )
    if ev.track_count < HIGH_TRACK_COUNT_MIN:
        return f"only {ev.track_count} tracks support the reconstruction."
    if ev.inlier_ratio < HIGH_INLIER_RATIO_MIN and ev.parallax < PARALLAX_OBSERVABLE_MIN:
        return (
            f"the dominant 2D motion model explained only {_percent(ev.inlier_ratio)} of "
            f"tracked features."
        )
    if ev.texture < TEXTURE_GOOD or ev.sharpness < SHARPNESS_GOOD:
        return (
            f"image quality limits the evidence (texture {ev.texture:.2f}, sharpness "
            f"{ev.sharpness:.2f})."
        )
    return "not every piece of evidence reached the level this system calls HIGH."
