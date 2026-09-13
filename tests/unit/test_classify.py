"""Motion labels on ground-truth Blender cameras (spec section 13)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.geometry.conventions import blender_quat_to_camerapath
from app.models.schemas.motion import LensFrame
from app.models.schemas.trajectory import CameraPose, ScaleMode
from app.trajectory.classify import classify_motion
from app.trajectory.kinematics import compute_kinematics

SYNTH = Path(__file__).resolve().parents[2] / "benchmarks" / "synthetic"

CASES = [
    ("pan", False, {"pan_right"}),
    ("tilt", False, {"tilt_up"}),
    ("roll", False, {"roll"}),
    ("orbit", True, {"orbit"}),
    ("dolly_forward", True, {"dolly_in"}),
    ("truck_right", True, {"truck_right"}),
    ("crane_up", True, {"rise_and_tilt"}),
    ("zoom_only", False, {"zoom_in"}),
    ("static", False, {"static"}),
    ("accelerating", True, {"dolly_in"}),
]


def _labels(scene: str, observable: bool, jitter: float = 0.0) -> tuple[set[str], str]:
    truth = SYNTH / f"{scene}.truth.json"
    if not truth.is_file():
        pytest.skip(f"synthetic scene {scene} not rendered (scripts/render_synthetic.sh)")
    frames = json.loads(truth.read_text())["frames"]
    poses = [CameraPose(frame_index=f["frame"], timestamp=f["time"], position=f["location"],
                        quaternion=[float(v) for v in blender_quat_to_camerapath(np.array(f["quaternion_wxyz"]))],
                        fov_horizontal=f["fov_horizontal"], focal_normalized=0.5) for f in frames]
    lens = [LensFrame(frame_index=p.frame_index, timestamp=p.timestamp, focal_normalized=0.5,
                      fov_horizontal=p.fov_horizontal, fov_vertical=40.0) for p in poses]
    moves, summary = classify_motion(poses, compute_kinematics(poses, ScaleMode.NORMALIZED), lens,
                                     jitter_score=jitter, translation_observable=observable)
    return {m.label.value for m in moves}, summary


@pytest.mark.parametrize("scene,observable,expected", CASES)
def test_ground_truth_moves_get_the_right_label(scene, observable, expected):
    labels, _ = _labels(scene, observable)
    assert labels == expected, f"{scene}: {labels}"


def test_translation_labels_are_suppressed_when_translation_was_not_observed():
    """I6/I7: a dolly whose translation was not measured must not be called a dolly."""
    labels, summary = _labels("dolly_forward", observable=False)
    assert not labels & {"dolly_in", "dolly_out", "truck_left", "truck_right",
                         "pedestal_up", "pedestal_down", "orbit", "arc", "crane"}
    assert "not observable" in summary


def test_orbit_is_not_described_as_a_roll():
    """A pitched camera orbiting picks up body-frame roll; operators would not call it one."""
    labels, _ = _labels("orbit", observable=True)
    assert "roll" not in labels


def test_handheld_is_flagged_from_jitter():
    labels, _ = _labels("handheld", observable=True, jitter=0.37)
    assert "handheld" in labels
