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


def estimate_focal_from_homographies(
    motion_frames: list[MotionFrame], width: int, height: int
) -> tuple[float | None, float]:
    """Estimate focal length from inter-frame homographies of a rotating camera.

    For a camera that only rotates, consecutive views are related by
    `H ~ K R K^-1`. That constrains K, and with a centred principal point and
    square pixels it leaves one unknown — the focal length.

    Returns (focal_pixels, confidence). None when the shot does not contain
    enough rotation to constrain anything: a pure translation gives a homography
    that is consistent with *any* focal length, so producing a number there
    would be fabrication.
    """
    candidates: list[float] = []

    for mf in motion_frames:
        if mf.homography is None or mf.inlier_ratio < 0.7:
            continue
        # Needs real rotation to be informative.
        if abs(mf.rotation_deg) < 0.05 and abs(mf.dx_pixels) < 1.0 and abs(mf.dy_pixels) < 1.0:
            continue

        h = np.array(mf.homography, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
            continue
        h = h / h[2, 2]

        # Shift to a principal-point-centred frame so K = diag(f, f, 1).
        cx, cy = width / 2.0, height / 2.0
        t = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        hc = t @ h @ np.linalg.inv(t)

        # With K = diag(f,f,1), H = K R K^-1 implies these two constraints on
        # f^2 (from orthonormality of R's first two columns).
        # R = K^-1 H K, and R^T R = I gives:
        #   f^2 = -(h00*h01 + h10*h11) / (h20*h21)     [column orthogonality]
        denom = hc[2, 0] * hc[2, 1]
        if abs(denom) > 1e-14:
            f_sq = -(hc[0, 0] * hc[0, 1] + hc[1, 0] * hc[1, 1]) / denom
            if np.isfinite(f_sq) and f_sq > 0:
                candidates.append(float(np.sqrt(f_sq)))

        # Second constraint from equal column norms.
        denom2 = hc[2, 0] ** 2 - hc[2, 1] ** 2
        if abs(denom2) > 1e-14:
            num2 = (hc[0, 1] ** 2 + hc[1, 1] ** 2) - (hc[0, 0] ** 2 + hc[1, 0] ** 2)
            f_sq = num2 / denom2
            if np.isfinite(f_sq) and f_sq > 0:
                candidates.append(float(np.sqrt(f_sq)))

    if len(candidates) < 8:
        return None, 0.0

    arr = np.array(candidates)
    # Keep physically plausible values only.
    lo = fov_to_focal_pixels(MAX_HORIZONTAL_FOV, width)
    hi = fov_to_focal_pixels(MIN_HORIZONTAL_FOV, width)
    arr = arr[(arr > lo) & (arr < hi)]
    if len(arr) < 8:
        return None, 0.0

    focal = float(np.median(arr))
    # Confidence from agreement: a wide spread means the constraints disagree,
    # which happens when the motion is not actually rotation-dominated.
    spread = float(np.median(np.abs(arr - focal)) / max(focal, 1e-6))
    confidence = float(np.clip(1.0 - spread * 4.0, 0.0, 0.75)) * min(1.0, len(arr) / 40.0)
    log.info(
        "focal from %d homography constraints: %.1f px (spread %.1f%%, confidence %.2f)",
        len(arr), focal, spread * 100, confidence,
    )
    return focal, confidence


def build_lens_curve(
    motion_frames: list[MotionFrame],
    base: CameraIntrinsics,
    *,
    parallax_score: float,
    lens_mode: str = "auto",
    smooth_window: int = 9,
) -> tuple[list[LensFrame], str]:
    """Per-frame focal/FOV curve.

    Zoom is detected from the accumulated image scale factor. The critical
    caveat is baked in: expanding radial flow is produced by a dolly-in AND by a
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

    long_edge = max(base.width, base.height)
    cumulative = np.cumprod([max(mf.scale, 1e-3) for mf in motion_frames])
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
    return curve, note


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
