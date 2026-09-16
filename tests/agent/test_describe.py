"""The honesty gates on the prompt-ready description.

The failure this module guards against is not an ugly sentence — it is a fluent
one that says the camera flew through a room when all the pipeline measured was
a pan. Each test below is one rule that keeps that from happening.
"""

from __future__ import annotations

import pytest

from app.agent import report as report_mod
from app.agent.describe import (
    NORMALIZED_CAVEAT,
    NOT_OBSERVABLE_SENTENCE,
    PERCEPTUAL_CAVEAT,
    PHRASES,
    TRANSLATION_LABELS,
    describe_job,
    describe_shot,
)
from app.models.schemas.trajectory import (
    ClassifiedMove,
    ConfidenceLevel,
    MotionLabel,
    ScaleMode,
    SolverSource,
)
from app.trajectory.classify import ZOOM_RATIO_THRESHOLD
from tests.agent.conftest import make_job, make_trajectory

METRIC_WORDS = (" m ", " m.", "metre", "meter", "feet", "foot")


def _shot(**kwargs):
    trajectory = make_trajectory(**kwargs)
    job = make_job(trajectories=[trajectory])
    return report_mod.job_report(job, ui_url="u", job_dir="d").shots[0]


@pytest.mark.parametrize("label", list(MotionLabel))
def test_every_motion_label_has_a_phrase(label: MotionLabel):
    # Parametrised over the enum so a label added to the pipeline fails the
    # suite instead of silently rendering as nothing.
    assert PHRASES[label].strip()


def test_no_translation_vocabulary_when_translation_is_not_observable():
    # Feed it every translation label anyway: classify_motion would not emit
    # these, and the description must not repeat them if something upstream did.
    moves = [
        ClassifiedMove(label=label, strength=0.9, start_time=0.0, end_time=1.0,
                       description="should never be rendered")
        for label in sorted(TRANSLATION_LABELS, key=lambda m: m.value)
    ]
    shot = _shot(translation_observable=False, moves=moves)
    text = describe_shot(shot)
    for label in TRANSLATION_LABELS:
        assert PHRASES[label] not in text, label
    assert NOT_OBSERVABLE_SENTENCE in text


def test_the_not_observable_caveat_sentence_is_present():
    shot = _shot(translation_observable=False)
    text, _caveats, derived = describe_job([shot])
    assert NOT_OBSERVABLE_SENTENCE in text
    assert "translation.observable" in derived


def test_rotation_still_gets_described_when_translation_is_not_observable():
    shot = _shot(
        translation_observable=False,
        moves=[ClassifiedMove(label=MotionLabel.PAN_LEFT, strength=0.8, start_time=0.0,
                              end_time=1.0, description="104 deg pan left")],
    )
    text = describe_shot(shot)
    assert "pans left" in text
    assert "104 deg pan left" in text


def test_normalized_scale_never_produces_metric_vocabulary():
    shot = _shot(scale_mode=ScaleMode.NORMALIZED)
    text, caveats, _ = describe_job([shot], scale_units="normalized")
    blob = " ".join([text, *caveats])
    for word in METRIC_WORDS:
        assert word not in blob, word
    assert NORMALIZED_CAVEAT in caveats


def test_metric_scale_does_not_add_the_normalized_caveat():
    shot = _shot(scale_mode=ScaleMode.METRIC, metric_scale_factor=2.0)
    _text, caveats, _ = describe_job([shot], scale_units="m")
    assert NORMALIZED_CAVEAT not in caveats


def test_perceptual_solver_adds_its_caveat():
    shot = _shot(solver=SolverSource.PERCEPTUAL)
    _text, caveats, derived = describe_job([shot])
    assert PERCEPTUAL_CAVEAT in caveats
    assert "solver_used" in derived


def test_low_confidence_adds_the_prefix_and_the_top_two_reasons():
    reasons = ["only 41% of frames registered", "median reprojection error 2.8 px", "third"]
    shot = _shot(level=ConfidenceLevel.LOW, reasons=reasons)
    text, caveats, _ = describe_job([shot])
    assert text.startswith("Low-confidence recovery: ")
    assert reasons[0] in caveats and reasons[1] in caveats
    assert reasons[2] not in caveats


def test_a_sub_threshold_fov_drift_is_never_called_a_zoom():
    drift = 60.0 / (ZOOM_RATIO_THRESHOLD ** 0.5)
    shot = _shot(fov_start=60.0, fov_end=drift)
    text = describe_shot(shot)
    assert "zoom" not in text.lower()
    assert shot.lens.focal_ratio is not None


def test_a_real_zoom_label_is_described():
    shot = _shot(
        moves=[ClassifiedMove(label=MotionLabel.ZOOM_IN, strength=0.5, start_time=0.0,
                              end_time=1.0, description="focal length x1.34")]
    )
    assert "zooms in" in describe_shot(shot)
    assert "focal length x1.34" in describe_shot(shot)


def test_magnitudes_come_verbatim_from_the_classifier():
    move = ClassifiedMove(label=MotionLabel.PAN_RIGHT, strength=0.6, start_time=0.0,
                          end_time=1.0, description="37 deg pan right")
    shot = _shot(moves=[move])
    assert move.description in describe_shot(shot)


def test_a_partial_move_gets_a_time_span_and_a_whole_shot_move_does_not():
    whole = _shot(moves=[ClassifiedMove(label=MotionLabel.PAN_LEFT, strength=0.5,
                                        start_time=0.0, end_time=1.0, description="d")])
    assert "(0.0-1.0 s)" not in describe_shot(whole)
    partial = _shot(moves=[ClassifiedMove(label=MotionLabel.PAN_LEFT, strength=0.5,
                                          start_time=0.2, end_time=0.5, description="d")])
    assert "(0.2-0.5 s)" in describe_shot(partial)


def test_pacing_comes_from_measured_speeds_only():
    fast_end = [1.0] * 20 + [8.0] * 10
    accelerating = _shot(frames=30, speeds=fast_end)
    assert "accelerating" in describe_shot(accelerating)

    fast_start = [8.0] * 10 + [1.0] * 20
    slowing = _shot(frames=30, speeds=fast_start)
    assert "slowing" in describe_shot(slowing)

    even = _shot(frames=30, speeds=[1.0] * 30)
    assert "accelerating" not in describe_shot(even)
    assert "slowing" not in describe_shot(even)


def test_no_pacing_when_translation_is_not_observable():
    shot = _shot(frames=30, speeds=[1.0] * 20 + [8.0] * 10, translation_observable=False)
    text = describe_shot(shot)
    assert "accelerating" not in text and "slowing" not in text


def test_multi_shot_joins_with_a_hard_cut_sentence_and_a_header():
    job = make_job(shots=2)
    shots = report_mod.job_report(job, ui_url="u", job_dir="d").shots
    text, _caveats, _ = describe_job(shots)
    assert "Hard cut." in text
    assert "2 shots separated by hard cuts" in text
    assert "own coordinate system" in text
    assert text.count("Shot ") >= 2


def test_max_chars_truncates_clauses_but_never_caveats():
    job = make_job(shots=3)
    shots = report_mod.job_report(job, ui_url="u", job_dir="d").shots
    for shot in shots:
        shot.confidence.level = "low"
        shot.confidence.reasons = ["a long reason that must survive truncation intact"]
    text, caveats, _ = describe_job(shots, max_chars=120)
    assert len(text) <= 120
    assert "a long reason that must survive truncation intact" in caveats


def test_brief_style_is_the_primary_move_and_the_duration():
    shot = _shot(moves=[ClassifiedMove(label=MotionLabel.ORBIT, strength=0.9, start_time=0.0,
                                       end_time=1.0, description="104 deg of yaw")])
    text = describe_shot(shot, style="brief")
    assert text.startswith("Orbits around the subject")
    assert text.endswith("s.")


def test_technical_style_appends_the_pipeline_summary_and_the_solver():
    shot = _shot()
    text = describe_shot(shot, style="technical")
    assert shot.pipeline_summary in text
    assert "Solver: colmap" in text
    assert "Confidence high" in text


def test_derived_from_is_an_audit_trail_without_duplicates():
    job = make_job(shots=2)
    shots = report_mod.job_report(job, ui_url="u", job_dir="d").shots
    _text, _caveats, derived = describe_job(shots)
    assert len(derived) == len(set(derived))
    assert "classified_moves" in derived
