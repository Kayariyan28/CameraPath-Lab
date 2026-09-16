"""The one test that has to use a real subprocess.

Under the stdio transport, file descriptor 1 *is* the JSON-RPC channel. Setting
`sys.stdout` is not enough to protect it: pycolmap is a C++ extension, and
Blender and ffmpeg are child processes, none of which can see a Python-level
attribute. Only a real process with real pipes can prove the fd-level swap
works, so that is what this does.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "cpl-mcp"
PROTOCOL_VERSION = "2025-06-18"

pytestmark = [pytest.mark.slow, pytest.mark.mcp]

mcp_available = pytest.importorskip("mcp") is not None

CONTAMINANTS = ("PRINT-CONTAMINATION", "FD1-CONTAMINATION", "LOG-CONTAMINATION",
                "CHILD-CONTAMINATION")

#: A server that registers one deliberately filthy tool, then runs the real
#: entrypoint. Nothing test-only lives in the product module.
NOISY_SERVER = '''
import logging, os, subprocess, sys
from app.agent import mcp_server

@mcp_server.mcp.tool()
def contaminate() -> str:
    """Write to stdout in every way a real solve could."""
    print("PRINT-CONTAMINATION")
    os.write(1, b"FD1-CONTAMINATION\\n")
    logging.getLogger("cpl.test").info("LOG-CONTAMINATION")
    subprocess.run(["sh", "-c", "echo CHILD-CONTAMINATION"], check=False)
    return "done"

sys.exit(mcp_server.main(["--runner", "local"]))
'''


class _Server:
    """A minimal newline-delimited JSON-RPC client over the server's pipes."""

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env, cwd=str(cwd),
        )
        self.stdout_lines: list[str] = []
        self.stderr_text: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            self.stdout_lines.append(line)
            self._queue.put(line)
        self._queue.put(None)

    def _pump_stderr(self) -> None:
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.stderr_text.append(line)

    def send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def response(self, request_id: int, timeout: float = 90.0) -> dict:
        while True:
            line = self._queue.get(timeout=timeout)
            if line is None:
                raise AssertionError(
                    "server closed stdout before answering; stderr:\n"
                    + "".join(self.stderr_text[-40:])
                )
            payload = json.loads(line)  # every stdout byte must be JSON-RPC
            if payload.get("id") == request_id:
                return payload

    def handshake(self) -> dict:
        self.send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "hygiene-test", "version": "0"},
            },
        })
        result = self.response(1)
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=10)


def _env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CPL_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "CPL_AGENT_RUNNER": "local",
        "PYTHONUNBUFFERED": "1",
    })
    env.update(extra or {})
    return env


@pytest.fixture()
def wrapper_server(tmp_path: Path):
    if not WRAPPER.is_file() or not os.access(WRAPPER, os.X_OK):
        pytest.skip("scripts/cpl-mcp is missing or not executable")
    server = _Server([str(WRAPPER)], _env(tmp_path), REPO_ROOT)
    try:
        yield server
    finally:
        server.close()


def test_the_wrapper_speaks_clean_json_rpc_over_stdio(wrapper_server: _Server):
    init = wrapper_server.handshake()
    assert init["result"]["serverInfo"]["name"] == "camerapath-lab"
    assert "normalized units, not metres" in init["result"]["instructions"]

    wrapper_server.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = wrapper_server.response(2)["result"]["tools"]
    assert "start_camera_motion_recovery" in {t["name"] for t in tools}

    # Every byte that reached stdout parsed as JSON-RPC, by construction of
    # `response`. Assert it again over the whole transcript.
    for line in wrapper_server.stdout_lines:
        if line.strip():
            json.loads(line)


def test_startup_logging_goes_to_stderr_not_stdout(wrapper_server: _Server):
    wrapper_server.handshake()
    wrapper_server.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    wrapper_server.response(2)
    stderr = "".join(wrapper_server.stderr_text)
    assert "camerapath-lab MCP server ready" in stderr
    assert "camerapath-lab MCP server ready" not in "".join(wrapper_server.stdout_lines)


def test_no_print_or_child_output_can_reach_the_protocol_stream(tmp_path: Path):
    script = tmp_path / "noisy_server.py"
    script.write_text(NOISY_SERVER)
    env = _env(tmp_path, {"PYTHONPATH": str(REPO_ROOT / "backend")})
    server = _Server([sys.executable, str(script)], env, REPO_ROOT)
    try:
        server.handshake()
        server.send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "contaminate", "arguments": {}},
        })
        assert server.response(2)["result"]["isError"] is False

        stdout = "".join(server.stdout_lines)
        stderr = "".join(server.stderr_text)
        for marker in CONTAMINANTS:
            assert marker not in stdout, f"{marker} reached the protocol stream"
            assert marker in stderr, f"{marker} vanished instead of going to stderr"

        # ...and the server is still healthy afterwards.
        server.send({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        assert server.response(3)["result"]["tools"]
    finally:
        server.close()


def test_no_module_in_the_agent_package_prints_except_the_cli():
    """`print` in the agent package is a stdio corruption waiting to happen.

    The fd swap catches it at runtime; this catches it at review time, and keeps
    `cli.py` as the single module allowed to write to stdout.
    """
    package = REPO_ROOT / "backend" / "app" / "agent"
    offenders = []
    for path in sorted(package.glob("*.py")):
        if path.name == "cli.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("print(") or " print(" in stripped:
                offenders.append(f"{path.name}:{number}: {stripped}")
            if "sys.stdout.write" in stripped:
                offenders.append(f"{path.name}:{number}: {stripped}")
    assert offenders == [], "stdout writes outside cli.py:\n" + "\n".join(offenders)
