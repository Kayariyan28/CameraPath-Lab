"""Prompt-ready natural-language description of a recovered camera move.

Written for a generative video model's conditioning text, which is why the
default style is `prompt`. Everything it can say comes from a `ShotMotionReport`
and nothing else: the classified moves and their own `description` strings, the
shot's duration, the observability verdict, the rotation total, the lens ratio,
the confidence level, the scale units and the solver that produced them.

The hard rule is that no number in the prose is computed here. A magnitude in
the text is always the classifier's own phrase, copied verbatim, so any clause
can be traced to a field. The honesty gates below exist because the failure this
module has to avoid is not an ugly sentence — it is a fluent one that says the
camera flew through a room when all that was measured was a pan.
"""

from __future__ import annotations

from app.agent.models import DescriptionStyle, ShotMotionReport
from app.models.schemas.trajectory import MotionLabel

#: One fixed phrase per label. Exhaustive by construction and unit-tested
#: against the enum, so a label added to the pipeline cannot silently render as
#: nothing — it fails the suite instead.
PHRASES: dict[MotionLabel, str] = {
    MotionLabel.STATIC: "locked off, static",
    MotionLabel.PAN_LEFT: "pans left",
    MotionLabel.PAN_RIGHT: "pans right",
    MotionLabel.TILT_UP: "tilts up",
    MotionLabel.TILT_DOWN: "tilts down",
    MotionLabel.ROLL: "rolls",
    MotionLabel.DOLLY_IN: "pushes forward",
    MotionLabel.DOLLY_OUT: "pulls back",
    MotionLabel.TRUCK_LEFT: "tracks left",
    MotionLabel.TRUCK_RIGHT: "tracks right",
    MotionLabel.PEDESTAL_UP: "rises straight up",
    MotionLabel.PEDESTAL_DOWN: "drops straight down",
    MotionLabel.CRANE: "cranes, rising while re-aiming",
    MotionLabel.ORBIT: "orbits around the subject",
    MotionLabel.ARC: "arcs around the subject",
    MotionLabel.PUSH_IN: "pushes in on the subject",
    MotionLabel.PULL_OUT: "pulls out from the subject",
    MotionLabel.HANDHELD: "handheld, with visible shake",
    MotionLabel.FPV: "flies forward, banking through the turns",
    MotionLabel.DRONE_FLY_THROUGH: "flies through the scene",
    MotionLabel.DRONE_REVEAL: "pulls back to reveal the scene",
    MotionLabel.RISE_AND_TILT: "rises while tilting down",
    MotionLabel.SPIRAL: "spirals around the subject",
    MotionLabel.ZOOM_IN: "zooms in",
    MotionLabel.ZOOM_OUT: "zooms out",
    MotionLabel.MIXED_6DOF: "moves on several axes at once",
}

#: Labels whose phrase asserts the camera moved through space. None of these may
#: appear when `translation.observable` is false — belt and braces, because
#: `classify_motion` will not have emitted them either (I6).
TRANSLATION_LABELS: frozenset[MotionLabel] = frozenset({
    MotionLabel.DOLLY_IN,
    MotionLabel.DOLLY_OUT,
    MotionLabel.TRUCK_LEFT,
    MotionLabel.TRUCK_RIGHT,
    MotionLabel.PEDESTAL_UP,
    MotionLabel.PEDESTAL_DOWN,
    MotionLabel.CRANE,
    MotionLabel.ORBIT,
    MotionLabel.ARC,
    MotionLabel.PUSH_IN,
    MotionLabel.PULL_OUT,
    MotionLabel.FPV,
    MotionLabel.DRONE_FLY_THROUGH,
    MotionLabel.DRONE_REVEAL,
    MotionLabel.RISE_AND_TILT,
    MotionLabel.SPIRAL,
})

#: At most this many clauses per shot. Beyond four the text stops reading as a
#: camera direction and starts reading as a data dump.
MAX_CLAUSES = 4

#: peak/mean speed ratio above which the shot's speed is visibly uneven.
SPEED_UNEVEN_RATIO = 1.8

#: Fraction of the shot within which a speed peak counts as "at the start" or
#: "at the end".
SPEED_PEAK_EDGE = 0.4

NOT_OBSERVABLE_SENTENCE = (
    "Translation was not observable in this shot, so only rotation and lens are described."
)

PERCEPTUAL_CAVEAT = (
    "recovered by screen-space Perceptual Match; this is a matched appearance, "
    "not a measured physical camera path."
)

NORMALIZED_CAVEAT = (
    "distances are relative (normalized units), not metric; monocular video "
    "carries no absolute scale."
)

LOW_CONFIDENCE_PREFIX = "Low-confidence recovery: "


def _clause_for(move, shot: ShotMotionReport) -> str | None:
    """One clause for one classified move, or None if it must be dropped."""
    try:
        label = MotionLabel(move.label)
    except ValueError:
        return None
    if label is MotionLabel.MIXED_6DOF:
        return None  # a meta-label about the other labels, not a move
    if not shot.translation.observable and label in TRANSLATION_LABELS:
        return None
    phrase = PHRASES[label]
    # The parenthetical is the classifier's own description, verbatim — that is
    # what keeps every number in the prose traceable to a measurement.
    text = f"{phrase} ({move.description})" if move.description else phrase
    span = move.end_time - move.start_time
    if shot.duration_seconds > 0 and span < 0.95 * shot.duration_seconds:
        text += f" ({move.start_time:.1f}-{move.end_time:.1f} s)"
    return text


def _pacing(shot: ShotMotionReport) -> str | None:
    """"accelerating" / "slowing", from measured speeds only.

    Deviation from the written spec, deliberately: it proposed
    `peak/mean <= 0.6` as the "slowing" test, which no sample can satisfy —
    a peak is never below its own mean. The direction of the change is instead
    read from *when* the peak happened, which is a measured field
    (`peak_speed_time`) and is the only thing in the report that distinguishes a
    move that sped up from one that slowed down.
    """
    t = shot.translation
    if not t.observable or not t.mean_speed or t.peak_speed is None:
        return None
    if t.mean_speed <= 0 or t.peak_speed_time is None:
        return None
    if (t.peak_speed / t.mean_speed) < SPEED_UNEVEN_RATIO:
        return None
    duration = shot.duration_seconds
    if duration <= 0:
        return None
    offset = (t.peak_speed_time - shot.start_time) / duration
    if offset <= SPEED_PEAK_EDGE:
        return "slowing"
    if offset >= 1.0 - SPEED_PEAK_EDGE:
        return "accelerating"
    return None


def describe_shot(shot: ShotMotionReport, *, style: DescriptionStyle = "prompt") -> str:
    """One paragraph for one shot."""
    if style == "brief":
        primary = shot.primary_move
        if primary:
            try:
                phrase = PHRASES[MotionLabel(primary)]
            except ValueError:
                phrase = primary
        else:
            phrase = PHRASES[MotionLabel.STATIC]
        return f"{phrase.capitalize()}, {shot.duration_seconds:.1f} s."

    moves = sorted(shot.moves, key=lambda m: (m.start_time, -m.strength))
    clauses: list[tuple[float, str]] = []
    for move in moves:
        clause = _clause_for(move, shot)
        if clause is not None:
            clauses.append((move.strength, clause))
        if len(clauses) >= MAX_CLAUSES:
            break

    if not clauses:
        body = PHRASES[MotionLabel.STATIC]
    else:
        body = ", then ".join(text for _, text in clauses)

    pace = _pacing(shot)
    if pace:
        body += f", {pace}"

    text = f"Camera {body}, over {shot.duration_seconds:.1f} s."

    if style == "technical":
        extras = [
            f"Solver: {shot.solver_used} ({shot.pipeline_mode_used}).",
            f"Confidence {shot.confidence.level} ({shot.confidence.score:.2f}).",
            (
                "Translation observable."
                if shot.translation.observable
                else NOT_OBSERVABLE_SENTENCE
            ),
        ]
        if shot.pipeline_summary:
            extras.append(shot.pipeline_summary)
        text = " ".join([text, *extras])
    elif not shot.translation.observable:
        text = f"{text} {NOT_OBSERVABLE_SENTENCE}"

    return text


def describe_job(
    shots: list[ShotMotionReport],
    *,
    style: DescriptionStyle = "prompt",
    max_chars: int = 600,
    scale_units: str = "normalized",
) -> tuple[str, list[str], list[str]]:
    """(text, caveats, derived_from) for a whole job.

    Caveats are returned separately and are never truncated: the reason a reader
    should distrust a sentence must not be the thing that falls off the end of
    it.
    """
    caveats: list[str] = []
    derived: list[str] = ["classified_moves", "duration_seconds"]

    paragraphs: list[str] = []
    for index, shot in enumerate(shots):
        body = describe_shot(shot, style=style)
        prefix = (
            f"Shot {index + 1} ({shot.start_time:.1f}-{shot.end_time:.1f} s): "
            if len(shots) > 1
            else ""
        )
        paragraphs.append(prefix + body)

        if shot.confidence.level == "low":
            paragraphs[-1] = prefix + LOW_CONFIDENCE_PREFIX + body[0].lower() + body[1:]
            for reason in shot.confidence.reasons[:2]:
                if reason not in caveats:
                    caveats.append(reason)
            derived.append("confidence.level")
        if shot.solver_used == "perceptual" and PERCEPTUAL_CAVEAT not in caveats:
            caveats.append(PERCEPTUAL_CAVEAT)
            derived.append("solver_used")
        if not shot.translation.observable:
            derived.append("translation.observable")
            if shot.translation.not_observable_reason:
                reason = shot.translation.not_observable_reason
                if reason not in caveats:
                    caveats.append(reason)
        else:
            derived.append("translation.total_path_length")
        if shot.rotation.total_rotation_deg:
            derived.append("rotation.total_rotation_deg")
        if shot.lens.focal_ratio is not None:
            derived.append("lens.focal_ratio")

    if scale_units == "normalized":
        caveats.append(NORMALIZED_CAVEAT)
        derived.append("scale.units")

    header = ""
    if len(shots) > 1:
        header = (
            f"{len(shots)} shots separated by hard cuts; each has its own "
            "coordinate system. "
        )
    text = header + " Hard cut. ".join(paragraphs)

    if len(text) > max_chars:
        # Trim whole shot paragraphs from the end rather than mangling a
        # sentence, and say plainly that something was dropped.
        kept = list(paragraphs)
        while len(kept) > 1 and len(header + " Hard cut. ".join(kept)) > max_chars:
            kept.pop()
        text = header + " Hard cut. ".join(kept)
        if len(kept) < len(paragraphs):
            caveats.append(
                f"description truncated to {len(kept)} of {len(paragraphs)} shots by "
                f"max_chars={max_chars}; call get_camera_motion for the rest."
            )
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 3)].rstrip() + "..."
            caveats.append(f"description truncated to max_chars={max_chars}; see caveats.")

    # Stable order, no duplicates — this list is an audit trail, not a set dump.
    seen: set[str] = set()
    derived_from = [d for d in derived if not (d in seen or seen.add(d))]
    return text, caveats, derived_from
