"""Guarded calls into OpenCV's robust estimators.

OpenCV's USAC/MAGSAC implementations can fail an internal assertion
(`!model.empty() in function 'setModelParameters'`) on a degenerate sample
instead of returning "no model". Measured: one handheld synthetic clip raised it
from `findFundamentalMat` partway through, and because these estimators run once
per frame, a single bad frame aborted the analysis of the entire shot (I12).

Every caller already handles a `None` model — that is the documented "could not
estimate" outcome — so an estimator error is mapped onto exactly that.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

import cv2

from app.core.logging import get_logger

log = get_logger("tracking.robust")

_failures: Counter[str] = Counter()
_lock = threading.Lock()


def robust(estimator: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Call a two-output OpenCV estimator; return (None, None) if it raises."""
    try:
        return estimator(*args, **kwargs)
    except cv2.error as exc:
        name = getattr(estimator, "__name__", "estimator")
        with _lock:
            _failures[name] += 1
            count = _failures[name]
        # Log the first few in full, then only occasionally: this can recur on
        # every frame of a pathological shot.
        if count <= 3 or count % 100 == 0:
            log.warning("%s failed (%d so far), treated as no model: %s",
                        name, count, str(exc).strip().splitlines()[-1][:160])
        return None, None


def failure_counts() -> dict[str, int]:
    with _lock:
        return dict(_failures)
