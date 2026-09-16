"""Input-path validation: what the agent surface will and will not open."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agent.models import AgentError
from app.agent.paths import ENV_ALLOWED_ROOTS, ENV_INPUT_CWD, attach_source, resolve_input_video
from app.config import Settings


def _video(tmp_path: Path, name: str = "clip.mp4", size: int = 2048) -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * size)
    return path


def _settings(tmp_path: Path, **kwargs) -> Settings:
    return Settings(workspace_dir=tmp_path / "workspace", **kwargs)  # type: ignore[call-arg]


def test_accepts_a_plain_video_file(tmp_path: Path):
    settings = _settings(tmp_path)
    video = _video(tmp_path)
    assert resolve_input_video(str(video), settings) == video.resolve()


def test_reports_the_resolved_absolute_path_for_a_relative_input(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    video = _video(tmp_path)
    monkeypatch.setenv(ENV_INPUT_CWD, str(tmp_path))
    assert resolve_input_video("clip.mp4", settings) == video.resolve()


@pytest.mark.parametrize("raw", ["", "   ", "a\x00b"])
def test_rejects_degenerate_strings(tmp_path: Path, raw: str):
    with pytest.raises(AgentError) as exc:
        resolve_input_video(raw, _settings(tmp_path))
    assert exc.value.code == "invalid_input_path"


def test_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(tmp_path / "nope.mp4"), _settings(tmp_path))
    assert exc.value.code == "invalid_input_path"


def test_rejects_a_directory(tmp_path: Path):
    (tmp_path / "dir.mp4").mkdir()
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(tmp_path / "dir.mp4"), _settings(tmp_path))
    assert exc.value.code == "invalid_input_path"


def test_rejects_an_unsupported_extension(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(path), _settings(tmp_path))
    assert exc.value.code == "unsupported_media_type"
    assert ".mp4" in exc.value.detail


def test_rejects_a_zero_byte_file(tmp_path: Path):
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(_video(tmp_path, size=0)), _settings(tmp_path))
    assert exc.value.code == "invalid_input_path"


def test_rejects_an_oversize_file(tmp_path: Path):
    settings = _settings(tmp_path, max_upload_bytes=16)
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(_video(tmp_path, size=64)), settings)
    assert exc.value.code == "file_too_large"


def test_rejects_traversal_that_lands_outside_an_allowlist_root(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _video(tmp_path, "outside.mp4")
    monkeypatch.setenv(ENV_ALLOWED_ROOTS, str(allowed))
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(allowed / ".." / "outside.mp4"), settings)
    assert exc.value.code == "invalid_input_path"
    assert str(allowed) in exc.value.hint
    # ...and the same file is fine once it is inside a root.
    inside = allowed / "inside.mp4"
    inside.write_bytes(outside.read_bytes())
    assert resolve_input_video(str(inside), settings) == inside.resolve()


def test_rejects_a_symlink_pointing_outside_an_allowlist_root(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = _video(tmp_path, "secret.mp4")
    link = allowed / "innocent.mp4"
    link.symlink_to(target)
    monkeypatch.setenv(ENV_ALLOWED_ROOTS, str(allowed))
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(link), settings)
    assert exc.value.code == "invalid_input_path"


def test_rejects_a_relative_path_escaping_the_input_cwd(tmp_path: Path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    settings = _settings(tmp_path)
    _video(tmp_path, "outside.mp4")
    monkeypatch.setenv(ENV_INPUT_CWD, str(base))
    monkeypatch.setenv(ENV_ALLOWED_ROOTS, str(base))
    with pytest.raises(AgentError) as exc:
        resolve_input_video("../outside.mp4", settings)
    assert exc.value.code == "invalid_input_path"


def test_rejects_a_path_inside_another_jobs_directory(tmp_path: Path):
    settings = _settings(tmp_path)
    other = settings.workspace_dir / "jobs" / "11111111-2222-3333-4444-555555555555" / "source"
    other.mkdir(parents=True)
    planted = other / "clip.mp4"
    planted.write_bytes(b"\0" * 32)
    with pytest.raises(AgentError) as exc:
        resolve_input_video(str(planted), settings)
    assert exc.value.code == "invalid_input_path"
    assert "workspace" in exc.value.message


def test_allows_a_jobs_own_source_when_named(tmp_path: Path):
    settings = _settings(tmp_path)
    job_id = "11111111-2222-3333-4444-555555555555"
    source = settings.workspace_dir / "jobs" / job_id / "source"
    source.mkdir(parents=True)
    clip = source / "clip.mp4"
    clip.write_bytes(b"\0" * 32)
    assert resolve_input_video(str(clip), settings, allow_job_id=job_id) == clip.resolve()


def test_rejects_an_unreadable_file(tmp_path: Path):
    video = _video(tmp_path)
    os.chmod(video, 0o000)
    try:
        with pytest.raises(AgentError) as exc:
            resolve_input_video(str(video), _settings(tmp_path))
        assert exc.value.code == "invalid_input_path"
    finally:
        os.chmod(video, 0o644)


def test_attach_hardlinks_by_default(tmp_path: Path):
    src = _video(tmp_path)
    dest = tmp_path / "job" / "source"
    target, how = attach_source(src, dest, mode="link")
    assert how == "link"
    assert target.stat().st_ino == src.stat().st_ino


def test_attach_falls_back_to_copy_when_linking_fails(tmp_path: Path, monkeypatch):
    src = _video(tmp_path)
    dest = tmp_path / "job" / "source"

    def boom(*_args, **_kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", boom)
    target, how = attach_source(src, dest, mode="link")
    assert how == "copy"
    assert target.read_bytes() == src.read_bytes()
    assert target.stat().st_ino != src.stat().st_ino


def test_attach_clears_a_previous_source(tmp_path: Path):
    dest = tmp_path / "job" / "source"
    dest.mkdir(parents=True)
    (dest / "old.mp4").write_bytes(b"old")
    target, _ = attach_source(_video(tmp_path, "new.mp4"), dest, mode="copy")
    assert [p.name for p in dest.iterdir()] == [target.name] == ["new.mp4"]
