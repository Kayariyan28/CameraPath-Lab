"""The glog threshold the agent front ends apply, and when they leave it alone."""

from __future__ import annotations

import logging

from app.agent.solver_logs import GLOG_ENV, QUIET_LEVEL, quiet_solver_logs


def test_quiets_the_solver_by_default(monkeypatch):
    monkeypatch.delenv(GLOG_ENV, raising=False)
    assert quiet_solver_logs(logging.INFO) is True
    import os

    assert os.environ[GLOG_ENV] == QUIET_LEVEL


def test_leaves_an_explicit_threshold_alone(monkeypatch):
    """Someone who set the variable has already chosen; do not override them."""
    monkeypatch.setenv(GLOG_ENV, "0")
    assert quiet_solver_logs(logging.INFO) is False
    import os

    assert os.environ[GLOG_ENV] == "0"


def test_debug_logging_keeps_the_solver_verbose(monkeypatch):
    """Debugging a solve is exactly when the solver's own account is wanted."""
    monkeypatch.delenv(GLOG_ENV, raising=False)
    assert quiet_solver_logs(logging.DEBUG) is False
    import os

    assert GLOG_ENV not in os.environ
