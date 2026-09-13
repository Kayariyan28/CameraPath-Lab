"""Blender runner: must never raise, must verify timing against the file itself,
and the render script's camera conversion must agree with the tested backend."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.blender import runner as runner_mod
from app.blender.runner import BlenderRunner, RenderResult, find_blender
from app.geometry.conventions import camerapath_quat_to_blender
from app.geometry.rotations import quat_from_axis_angle, quat_multiply

requires_blender = pytest.mark.skipif(find_blender() is None, reason="Blender not installed")


def _trajectory(path: Path, frames: int, fps: float) -> Path:
    """A dolly with a slow yaw and a tilt: exercises every rotation axis."""
    doc = {
        "video": {"filename": "synthetic"}, "coordinate_system": {"name": "camerapath_world"},
        "scale_mode": "normalized", "fps": fps, "frames": [],
    }
    for i in range(frames):
        q = quat_multiply(
            quat_from_axis_angle(np.array([0, 0, 1.0]), 0.02 * i),
            quat_from_axis_angle(np.array([1.0, 0, 0]), -0.1),
        )
        doc["frames"].append({
            "frame": i, "time": i / fps, "position": [0.0, 0.4 * i, 1.5],
            "quaternion": [float(v) for v in q], "fov": 55.0,
        })
    path.write_text(json.dumps(doc))
    return path


def test_find_blender_returns_an_executable_or_none():
    found = find_blender()
    assert found is None or Path(found).is_file()


def test_missing_blender_is_a_failed_result_not_an_exception(tmp_path):
    r = BlenderRunner()
    r.blender_path = None
    res = r.render_motion_proxy(_trajectory(tmp_path / "t.json", 4, 24), tmp_path / "o.mp4")
    assert isinstance(res, RenderResult) and not res.success and "Blender" in res.error


def test_malformed_trajectory_is_a_failed_result(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    res = BlenderRunner().render_motion_proxy(bad, tmp_path / "o.mp4")
    assert not res.success and "unreadable" in res.error
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"fps": 24, "frames": []}))
    assert not BlenderRunner().render_motion_proxy(empty, tmp_path / "o.mp4").success


def test_timing_mismatch_fails_loudly(tmp_path, monkeypatch):
    """I1: if the encoder wrote the wrong number of frames, that is a failure
    even though Blender itself exited cleanly."""
    out = tmp_path / "o.mp4"
    out.write_bytes(b"x")
    monkeypatch.setattr(runner_mod, "probe_mp4", lambda p: (59, 30.0, 1.97, 320, 180))
    res = BlenderRunner()._verify(out, expected_frames=60, expected_fps=30.0,
                                  result=RenderResult(False, output_path=out))
    assert not res.success
    assert "59 frames" in res.error and "FAILED" in res.error


def test_fps_mismatch_fails_loudly(tmp_path, monkeypatch):
    out = tmp_path / "o.mp4"
    out.write_bytes(b"x")
    monkeypatch.setattr(runner_mod, "probe_mp4", lambda p: (60, 25.0, 2.4, 320, 180))
    res = BlenderRunner()._verify(out, expected_frames=60, expected_fps=30.0,
                                  result=RenderResult(False, output_path=out))
    assert not res.success and "fps" in res.error


@requires_blender
@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.parametrize("fps", [24.0, 24000 / 1001])
def test_render_has_exact_frame_count_and_rate(tmp_path, fps):
    """End-to-end Phase 3 proof: a real render whose file timing matches exactly.
    23.976 is included because a rational rate is where rounding bugs live."""
    traj = _trajectory(tmp_path / "t.json", 24, fps)
    res = BlenderRunner().render_motion_proxy(
        traj, tmp_path / "proxy.mp4", width=320, height=180, samples=1,
        blend_path=tmp_path / "scene.blend",
    )
    assert res.success, res.error
    assert res.frame_count == 24
    assert res.fps == pytest.approx(fps, rel=1e-4)
    assert res.duration_seconds == pytest.approx(24 / fps, abs=1 / fps)
    assert (tmp_path / "scene.blend").is_file()


@requires_blender
@pytest.mark.blender
@pytest.mark.slow
def test_render_script_camera_matches_backend_convention(tmp_path):
    """I8: the render script reimplements the CameraPath->Blender quaternion
    conversion (Blender cannot import the backend). Its keyed cameras must agree
    with the tested backend function."""
    traj = _trajectory(tmp_path / "t.json", 12, 24.0)
    dump = tmp_path / "keyed.json"
    proc = subprocess.run(
        [find_blender(), "--background", "--factory-startup", "--python-exit-code", "3",
         "--python", str(runner_mod.SCRIPTS_DIR / "render_motion_proxy.py"), "--",
         "--trajectory", str(traj), "--out", str(tmp_path / "x.mp4"),
         "--no-render", "--dump-poses", str(dump)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    keyed = json.loads(dump.read_text())["frames"]
    source = json.loads(traj.read_text())["frames"]
    assert len(keyed) == len(source)
    for k, s in zip(keyed, source):
        expected = camerapath_quat_to_blender(np.array(s["quaternion"]))
        got = np.array(k["quaternion_wxyz"])
        # Stable angle (float32 storage inside Blender).
        dot = abs(float(np.dot(expected / np.linalg.norm(expected), got / np.linalg.norm(got))))
        assert dot > 1 - 1e-10, f"frame {s['frame']}: keyed camera disagrees with backend"
        assert np.allclose(k["location"], s["position"], atol=1e-5)
