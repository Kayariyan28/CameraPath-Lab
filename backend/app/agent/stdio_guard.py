"""File-descriptor level stdout hygiene for the MCP stdio transport.

Under stdio, file descriptor 1 *is* the JSON-RPC channel. One stray byte on it —
a log line, a `print`, a C++ extension writing to `stdout`, a child process that
inherited the descriptor — corrupts the stream and the client reports a parse
error with no clue where it came from.

Setting `sys.stdout` is not enough, and that is the whole reason this module
exists. `pycolmap` is a C++ extension; Blender and ffmpeg are child processes.
None of them can see a Python-level attribute. So fd 1 is replaced with a
duplicate of fd 2 before any backend module is imported, and the real stdout is
kept as a private duplicated descriptor that nothing can reach by name. After
that, everything that writes to "stdout" — at any level of the stack — lands on
stderr, where it is harmless and still visible to the user.
"""

from __future__ import annotations

import io
import logging
import os
import sys

#: Set once `install()` has run, so a second call is a no-op rather than a
#: second dup that would leak a descriptor.
_protocol_fd: int | None = None


def installed() -> bool:
    return _protocol_fd is not None


def install(level: int = logging.INFO) -> int:
    """Swap fd 1 to stderr and return the private descriptor for the protocol.

    Returns the duplicated original stdout descriptor. Call before importing
    anything that might log or spawn a subprocess.
    """
    global _protocol_fd
    if _protocol_fd is not None:
        return _protocol_fd

    from app.core.logging import configure_agent_logging

    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    # Python-level writes follow the descriptor swap, so `print()` anywhere in
    # the process — including code that has already captured `sys.stdout` — is
    # harmless.
    sys.stdout = sys.stderr

    configure_agent_logging(sys.stderr, level=level)

    _protocol_fd = protocol_fd
    return protocol_fd


def protocol_text_stream():
    """The JSON-RPC side of the swap, as an anyio-wrapped text stream.

    `FastMCP.run_stdio_async()` calls `stdio_server()` with no arguments, which
    would grab `sys.stdout.buffer` — now stderr. The entrypoint therefore runs
    the server loop itself and hands the transport this stream instead.
    """
    import anyio

    if _protocol_fd is None:
        raise RuntimeError("stdio_guard.install() has not run")
    raw = os.fdopen(_protocol_fd, "wb", buffering=0)
    return anyio.wrap_file(io.TextIOWrapper(raw, encoding="utf-8", write_through=True))
