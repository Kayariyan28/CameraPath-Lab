"""Agent-facing integration: one service layer, two front doors.

`AgentService` (service.py) is the single API. The MCP server (mcp_server.py)
and the JSON CLI (cli.py) are thin transports over it, so a tool call and a
shell invocation return byte-identical payloads for the same query.

This package is deliberately import-light: importing `app.agent` must not pull
in `pycolmap`, `cv2` or `mcp`. The heavy imports live inside the modules that
need them (`mcp` only in mcp_server.py, the solve stack only behind
`deps.Services.solve_pipeline()`), because an MCP client launches this process
over stdio and pays for every second of import time before the first tool call.
"""

from __future__ import annotations

__all__ = ["__doc__"]
