"""Turn down COLMAP's native logging for the agent front ends.

`pycolmap` is a C++ extension that logs through glog, straight to the process's
stderr — Python's logging configuration cannot see it. On a single orbit shot it
emits about 300 lines of per-image feature and registration detail, which buries
the six progress lines the CLI actually reports and makes the tool look broken
to anyone reading the output.

The web app is unaffected: there, the same chatter lands in the server's own
terminal, well away from the user, and the job log keeps the readable summary.
So this is applied at the agent entry points rather than in the pipeline, which
both front ends share with the web app.

glog reads its flags from the environment when it initialises, so setting the
variable before `pycolmap` is first imported is enough — and cheaper than
importing it here just to set an attribute.

Measured on one orbit shot: about 300 lines down to six. Not zero — some COLMAP
stages re-initialise glog and print a few INFO lines regardless — but those six
are the useful ones (bundle adjustment ran, how long it took), and the pipeline's
own log still reports how many keyframes registered and at what reprojection
error.
"""

from __future__ import annotations

import logging
import os

#: glog's own name for the threshold. Honoured if the caller already set it.
GLOG_ENV = "GLOG_minloglevel"

#: Warnings and errors only.
QUIET_LEVEL = "2"


def quiet_solver_logs(level: int = logging.INFO) -> bool:
    """Silence per-image solver chatter unless the user asked to see it.

    Returns whether the threshold was applied. Left alone when the caller has
    already set `GLOG_minloglevel`, or when logging at DEBUG — someone
    debugging a solve wants the solver's own account of it.
    """
    if GLOG_ENV in os.environ:
        return False
    if level <= logging.DEBUG:
        return False
    os.environ[GLOG_ENV] = QUIET_LEVEL
    return True
