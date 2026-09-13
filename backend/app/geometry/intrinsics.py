"""Camera intrinsics and the lens (focal/FOV) curve.

Two things this module refuses to do:

  * treat container metadata as truth. Metadata is a *prior*. Phone and drone
    exports routinely report the sensor's nominal focal length while the
    recorded stream is cropped, digitally stabilised, or shot in a different
    aspect — all of which change the effective field of view. So metadata seeds
    the estimate and is then tested against the imagery (spec §7).
  * assume constant focal length. A zoom is a real camera move and must survive
    into the output, so the lens is a per-frame curve, not a scalar (spec §7).

The hard problem here is that a dolly-in and a zoom-in produce the *same*
expanding radial flow. They are only separable by parallax: pushing the camera
forward changes the relative spacing of near and far content, while zooming
scales everything uniformly. Where parallax is absent the two are genuinely
indistinguishable, and this module says so rather than guessing (spec §34).
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.models.schemas.motion import CameraIntrinsics, LensFrame, MotionFrame
from app.models.schemas.video import VideoInfo

log = get_logger("geometry.intrinsics")

#: Default horizontal FOV prior when nothing better is known, in degrees.
#: 60 degrees corresponds to roughly a 35 mm lens on full frame — a reasonable
#: middle for cinematic and drone footage, and deliberately not a claim.
DEFAULT_HORIZONTAL_FOV = 60.0

#: Plausible bounds. Outside these an estimate is rejected as unphysical
#: rather than propagated: ~8 degrees is a long telephoto, ~150 an extreme
#: fisheye-ish wide.
MIN_HORIZONTAL_FOV = 8.0
MAX_HORIZONTAL_FOV = 150.0


def fov_to_focal_pixels(fov_degrees: float, sensor_pixels: int) -> float:
    """Pinhole focal length in pixels from a field of view across that axis."""
    half = np.radians(max(min(fov_degrees, 179.0), 0.1)) / 2.0
    return float(sensor_pixels / (2.0 * np.tan(half)))


def focal_pixels_to_fov(focal_pixels: float, sensor_pixels: int) -> float:
    if focal_pixels <= 0:
        return DEFAULT_HORIZONTAL_FOV
    return float(np.degrees(2.0 * np.arctan(sensor_pixels / (2.0 * focal_pixels))))


def intrinsics_from_fov(
    width: int, height: int, horizontal_fov_degrees: float, *,
    source: str = "estimated", confidence: float = 0.3,
) -> CameraIntrinsics:
    """Build intrinsics from a horizontal FOV, principal point at the centre.

    The centred principal point is an assumption, stated as such: real lenses
    are decentred by a percent or two, but estimating it from a monocular
    sequence without a calibration target is unreliable enough that a wrong
    estimate is worse than the assumption.
    """
    fx = fov_to_focal_pixels(horizontal_fov_degrees, width)
    # Square pixels. True for essentially all modern digital video; anamorphic
    # sources would need the sample aspect ratio applied here.
    fy = fx
    long_edge = max(width, height)
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
        focal_normalized=fx / long_edge,
        fov_horizontal=horizontal_fov_degrees,
        fov_vertical=focal_pixels_to_fov(fy, height),
        distortion=[],
        confidence=confidence,
        source=source,
    )


def _fov_from_metadata(info: VideoInfo) -> tuple[float, float, str] | None:
    """Try to read a credible FOV prior out of container tags.

    Returns (fov_degrees, confidence, note) or None. Confidence is capped well
    below 1.0 on purpose — see the module docstring on why metadata is a prior.
    """
    tags = {k.lower(): v for k, v in (info.metadata_tags or {}).items()}

    # 35 mm equivalent focal length is the only tag that translates directly
    # into an angle without knowing the sensor.
    for key in ("focallengthin35mmfilm", "focallengthin35mmformat", "com.apple.quicktime.focallength35mm"):
        raw = tags.get(key)
        if not raw:
            continue
        try:
            f35 = float(str(raw).split()[0])
        except (ValueError, IndexError):
            continue
        if not (8.0 <= f35 <= 400.0):
            continue
        # Full-frame width is 36 mm. The recorded frame may be cropped from the
        # sensor, so this stays a prior rather than a measurement.
        fov = float(np.degrees(2.0 * np.arctan(36.0 / (2.0 * f35))))
        if MIN_HORIZONTAL_FOV <= fov <= MAX_HORIZONTAL_FOV:
            return fov, 0.5, f"metadata 35mm-equivalent focal length {f35:.0f}mm"

    for key in ("cameraframereadouttime", "fieldofview", "com.dji.fov"):
        raw = tags.get(key)
        if not raw:
            continue
        try:
            fov = float(str(raw).split()[0])
        except (ValueError, IndexError):
            continue
        if MIN_HORIZONTAL_FOV <= fov <= MAX_HORIZONTAL_FOV:
            return fov, 0.45, f"metadata field of view {fov:.0f} degrees"

    return None


def initial_intrinsics(
    info: VideoInfo,
    width: int,
    height: int,
    *,
    fov_override_degrees: float | None = None,
) -> tuple[CameraIntrinsics, str]:
    """Best starting intrinsics for a shot, plus a human-readable provenance.

    Priority: explicit user override > credible metadata prior > default prior.
    The provenance string is surfaced in the UI so the user can see whether the
    FOV was measured, read, or assumed.
    """
    if fov_override_degrees is not None:
        fov = float(np.clip(fov_override_degrees, MIN_HORIZONTAL_FOV, MAX_HORIZONTAL_FOV))
        return (
            intrinsics_from_fov(width, height, fov, source="user_override", confidence=0.9),
            f"field of view set by user to {fov:.1f} degrees",
        )

    from_meta = _fov_from_metadata(info)
    if from_meta is not None:
        fov, conf, note = from_meta
        return (
            intrinsics_from_fov(width, height, fov, source="metadata_prior", confidence=conf),
            f"{note} (used as a prior, not as truth)",
        )

    return (
        intrinsics_from_fov(
            width, height, DEFAULT_HORIZONTAL_FOV, source="default_prior", confidence=0.2
        ),
        f"no lens metadata found; assuming {DEFAULT_HORIZONTAL_FOV:.0f} degrees horizontal "
        "as a starting prior",
    )


#: A chain of consecutive homographies is closed once it spans this many frames.
#: Per-frame rotation is often ~1 deg, too small to constrain focal; chaining
#: builds a baseline of 10+ deg while keeping accumulated error small.
FOCAL_CHAIN_MAX_FRAMES = 20
FOCAL_CHAIN_MIN_FRAMES = 3
FOCAL_CHAIN_MIN_INLIERS = 0.6

#: Structure cost at +-15% focal must exceed the cost at the optimum by this
#: factor for focal to count as measured. Calibrated on synthetic ground truth:
#: pan 10929 and tilt 3070 (both recovered to within 0.03 deg); roll, zoom-only
#: and static 1.00-1.01, which is the correct answer — K commutes with a roll
#: about the optical axis and with a zoom, so neither reveals focal length.
FOCAL_SHARPNESS_THRESHOLD = 10.0


def _homography_chains(motion_frames: list[MotionFrame]) -> list[np.ndarray]:
    chains: list[np.ndarray] = []
    current, length = np.eye(3), 0
    for mf in motion_frames:
        usable = (
            mf.homography is not None and len(mf.homography) == 9
            and mf.inlier_ratio >= FOCAL_CHAIN_MIN_INLIERS
        )
        h = np.asarray(mf.homography, dtype=np.float64).reshape(3, 3) if usable else None
        if h is None or not np.isfinite(h).all():
            if length >= FOCAL_CHAIN_MIN_FRAMES:
                chains.append(current)
            current, length = np.eye(3), 0
            continue
        current = h @ current
        length += 1
        if length >= FOCAL_CHAIN_MAX_FRAMES:
            chains.append(current)
            current, length = np.eye(3), 0
    if length >= FOCAL_CHAIN_MIN_FRAMES:
        chains.append(current)
    return chains


def _rotation_structure_cost(chains: list[np.ndarray], focal: float, width: int, height: int) -> float:
    """How far K^-1 H K is from (zoom) x (rotation), averaged over chains.

    A camera that rotates and zooms about the principal point gives
    M = K^-1 H K = diag(s,s,1) R, so M M^T = diag(s^2, s^2, 1) up to scale:
    zero off-diagonals and equal leading entries. That form holds at the true
    focal whatever the zoom, so a zoom cannot bias the estimate.
    """
    k = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    k_inv = np.linalg.inv(k)
    total = 0.0
    for h in chains:
        m = k_inv @ h @ k
        sym = m @ m.T
        trace = float(np.trace(sym))
        if trace <= 1e-12 or not np.isfinite(trace):
            continue
        sym = sym / (trace / 3.0)
        total += 2.0 * (sym[0, 1] ** 2 + sym[0, 2] ** 2 + sym[1, 2] ** 2) + (sym[0, 0] - sym[1, 1]) ** 2
    return total / max(len(chains), 1)


def estimate_focal_from_homographies(
    motion_frames: list[MotionFrame], width: int, height: int
) -> tuple[float | None, float]:
    """Focal length from the homographies of a camera that rotates (and zooms).

    Returns (focal_pixels, confidence) at the resolution the homographies were
    measured at, or (None, 0.0) when the shot does not constrain focal.

    Method: chain consecutive homographies into baselines of up to
    FOCAL_CHAIN_MAX_FRAMES, then find the focal that makes every chain look like
    a zoom times a rotation (`_rotation_structure_cost`), by grid search over the
    plausible FOV range and golden-section refinement. Observability is then
    TESTED, not assumed: the cost must rise sharply either side of the optimum.

    This replaced per-frame closed-form constraints that divided by products of
    tiny perspective terms. On a synthetic pan they returned 59.9 deg at
    confidence 0.63 against a true 65.5 deg; this method returns 65.50.

    Returns None — never a number — for pure roll, pure zoom and static shots,
    where focal is genuinely unobservable, and for shots whose translation makes
    the homography model itself invalid (use the geometric solve's focal there).
    """
    chains = _homography_chains(motion_frames)
    if len(chains) < 2 or width <= 0 or height <= 0:
        return None, 0.0

    fovs = np.arange(MIN_HORIZONTAL_FOV, MAX_HORIZONTAL_FOV + 1e-9, 0.5)
    focals = np.array([fov_to_focal_pixels(float(f), width) for f in fovs])
    costs = np.array([_rotation_structure_cost(chains, float(f), width, height) for f in focals])
    if not np.isfinite(costs).any():
        return None, 0.0
    i = int(np.nanargmin(costs))

    # Golden-section refinement between the neighbouring grid points.
    lo = focals[min(i + 1, len(focals) - 1)]  # focals decrease as FOV increases
    hi = focals[max(i - 1, 0)]
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = min(lo, hi), max(lo, hi)
    for _ in range(40):
        c = b - golden * (b - a)
        d = a + golden * (b - a)
        if _rotation_structure_cost(chains, c, width, height) < _rotation_structure_cost(chains, d, width, height):
            b = d
        else:
            a = c
    focal = 0.5 * (a + b)
    best = _rotation_structure_cost(chains, focal, width, height)

    sharpness = min(
        _rotation_structure_cost(chains, focal * 0.85, width, height),
        _rotation_structure_cost(chains, focal * 1.15, width, height),
    ) / (best + 1e-12)
    if not np.isfinite(sharpness) or sharpness < FOCAL_SHARPNESS_THRESHOLD:
        log.info("focal not observable from homographies (sharpness %.2f over %d chains)",
                 sharpness, len(chains))
        return None, 0.0

    confidence = float(np.clip(0.5 + 0.1 * np.log10(sharpness), 0.5, 0.9))
    log.info("focal from %d homography chains: %.1f px = %.2f deg (sharpness %.0f, confidence %.2f)",
             len(chains), focal, focal_pixels_to_fov(focal, width), sharpness, confidence)
    return float(focal), confidence


def zoom_factor_from_homography(homography: list[float] | None, base: CameraIntrinsics) -> float:
    """Per-transition zoom factor from a frame-to-frame homography; 1.0 if none.

    Why not the similarity `scale`: fitting a similarity to the perspective flow
    of a panning wide lens produces a large, steady, entirely spurious scale.
    Measured on a synthetic pure pan with a 65 deg lens: 1.0196 per frame — the
    same as a genuine 1.9%-per-frame zoom (1.0191). Accumulated over 59 frames it
    reported a 212% zoom and shrank the FOV from 60 to 21 deg.

    The intrinsics-normalised homography separates them. A camera that only
    rotates and zooms about the principal point has H = K diag(s,s,1) R K^-1, so
    K^-1 H K = diag(s,s,1) R, whose singular values are (s, s, 1) up to the
    homography's arbitrary scale — R contributes nothing. Pure rotation gives
    three equal values; a zoom scales two of them. Measured: pan [1.0001, 1.0001,
    1], zoom [1.0192, 1.0191, 1]. For a zoom the recovered factor does not even
    depend on the assumed focal, because diag(s,s,1) commutes with diag(f,f,1).

    The two closest singular values are the zoomed pair and the third is the
    unscaled axis, which handles zoom-out (pair below the odd one) as well as
    zoom-in.
    """
    if homography is None or len(homography) != 9:
        return 1.0
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(h).all():
        return 1.0
    k = np.array([[base.fx, 0.0, base.cx], [0.0, base.fy, base.cy], [0.0, 0.0, 1.0]])
    try:
        normalised = np.linalg.inv(k) @ h @ k
        sv = np.linalg.svd(normalised, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    if sv[-1] <= 1e-12:
        return 1.0
    if abs(sv[0] - sv[1]) <= abs(sv[1] - sv[2]):
        factor = float(np.sqrt(sv[0] * sv[1]) / sv[2])
    else:
        factor = float(np.sqrt(sv[1] * sv[2]) / sv[0])
    return factor if np.isfinite(factor) and factor > 0 else 1.0


def build_lens_curve(
    motion_frames: list[MotionFrame],
    base: CameraIntrinsics,
    *,
    parallax_score: float,
    lens_mode: str = "auto",
    smooth_window: int = 9,
) -> tuple[list[LensFrame], str]:
    """Per-frame focal/FOV curve.

    Zoom is detected from the accumulated zoom factor of the intrinsics-normalised
    frame-to-frame homographies (see `zoom_factor_from_homography` for why not
    the similarity scale). `base` must be at the analysis resolution the
    homographies were measured at. The critical caveat is baked in: expanding radial flow is produced by a dolly-in AND by a
    zoom-in, and they are only separable by parallax. So:

      * `lens_mode="fixed"`   — trust the user, emit a flat curve
      * `lens_mode="variable"`— attribute scale change to the lens
      * `lens_mode="auto"`    — attribute it to the lens only where parallax is
                                too low for a dolly to be the explanation, and
                                say which was assumed

    Genuine estimation noise is smoothed; a deliberate zoom is not (spec §7).
    """
    if not motion_frames:
        return [], "no frames to estimate a lens curve from"

    focal_note = ""
    # A rotating camera reveals its focal length; that measurement beats a prior.
    # Only below the parallax gate, where the homography model is valid, and
    # never over a user override.
    if base.source != "user_override" and parallax_score < 0.15:
        measured, measured_conf = estimate_focal_from_homographies(motion_frames, base.width, base.height)
        if measured is not None:
            fov = focal_pixels_to_fov(measured, base.width)
            base = intrinsics_from_fov(base.width, base.height, fov, source="estimated",
                                       confidence=measured_conf)
            focal_note = f"focal measured from camera rotation ({fov:.1f} deg horizontal); "

    long_edge = max(base.width, base.height)
    cumulative = np.cumprod([
        max(zoom_factor_from_homography(mf.homography, base), 1e-3) for mf in motion_frames
    ])
    net_change = float(cumulative[-1])

    # A real zoom moves the scale by more than tracking noise can explain.
    scale_is_significant = abs(net_change - 1.0) > 0.04

    if lens_mode == "fixed":
        attribute_to_lens = False
        note = "lens locked by user; all scale change attributed to camera translation"
    elif lens_mode == "variable":
        attribute_to_lens = scale_is_significant
        note = (
            f"lens set to variable; {abs(net_change - 1.0) * 100:.1f}% scale change "
            "attributed to zoom"
            if scale_is_significant else
            "lens set to variable but no significant scale change was measured"
        )
    else:
        # AUTO. Low parallax means a dolly cannot be distinguished from a zoom,
        # and in that regime a zoom is the safer attribution: it reproduces the
        # perceived move without inventing a physical baseline (spec §34).
        if not scale_is_significant:
            attribute_to_lens = False
            note = "no significant scale change; focal length treated as constant"
        elif parallax_score < 0.15:
            attribute_to_lens = True
            note = (
                f"{abs(net_change - 1.0) * 100:.1f}% scale change with parallax "
                f"{parallax_score:.2f} — too little depth variation to distinguish a "
                "dolly from a zoom, so it is attributed to the lens rather than "
                "inventing forward translation"
            )
        else:
            attribute_to_lens = False
            note = (
                f"{abs(net_change - 1.0) * 100:.1f}% scale change with parallax "
                f"{parallax_score:.2f} — depth-dependent flow indicates real forward "
                "motion, so it is attributed to a dolly, not a zoom"
            )

    base_focal = base.fx
    if attribute_to_lens:
        # Image content growing by factor s corresponds to focal length growing
        # by the same factor.
        focal_series = base_focal * cumulative
        focal_series = _smooth_preserving_trend(focal_series, smooth_window)
    else:
        focal_series = np.full(len(motion_frames), base_focal)

    lo = fov_to_focal_pixels(MAX_HORIZONTAL_FOV, base.width)
    hi = fov_to_focal_pixels(MIN_HORIZONTAL_FOV, base.width)
    focal_series = np.clip(focal_series, lo, hi)

    curve: list[LensFrame] = []
    # The curve covers output frames, so it needs one more entry than there are
    # transitions: the first frame has no preceding transition.
    first = LensFrame(
        frame_index=max(0, motion_frames[0].frame_index - 1),
        timestamp=max(0.0, motion_frames[0].timestamp - motion_frames[0].dt),
        focal_normalized=base_focal / long_edge,
        fov_horizontal=focal_pixels_to_fov(base_focal, base.width),
        fov_vertical=focal_pixels_to_fov(base_focal, base.height),
        confidence=base.confidence,
        is_estimated=base.source != "user_override",
    )
    curve.append(first)

    for mf, f in zip(motion_frames, focal_series):
        curve.append(
            LensFrame(
                frame_index=mf.frame_index,
                timestamp=mf.timestamp,
                focal_normalized=float(f) / long_edge,
                fov_horizontal=focal_pixels_to_fov(float(f), base.width),
                fov_vertical=focal_pixels_to_fov(float(f), base.height),
                confidence=base.confidence * (0.8 if attribute_to_lens else 1.0),
                is_estimated=True,
            )
        )
    return curve, focal_note + note


def _smooth_preserving_trend(series: np.ndarray, window: int) -> np.ndarray:
    """Median-then-mean smoothing that keeps a deliberate ramp intact.

    A plain moving average would flatten the start and end of a zoom, shortening
    it. The median pass removes per-frame outliers, and the short mean pass
    removes residual jitter, while both preserve a monotonic ramp's slope.
    """
    n = len(series)
    if n < 5 or window < 3:
        return series.copy()
    w = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
    if w < 3:
        return series.copy()
    half = w // 2
    padded = np.pad(series, half, mode="edge")
    med = np.array([np.median(padded[i:i + w]) for i in range(n)])
    w2 = max(3, w // 3)
    if w2 % 2 == 0:
        w2 += 1
    half2 = w2 // 2
    padded2 = np.pad(med, half2, mode="edge")
    kernel = np.ones(w2) / w2
    return np.convolve(padded2, kernel, mode="valid")[:n]
