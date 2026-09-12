"""Structured stage logging with a live subscriber fan-out.

Two consumers:
  * the job's `job.log` on disk (post-mortem)
  * the SSE endpoint (live progress in the UI)

Solver decisions are logged through here too, which is how invariant I9 ("log
solver decisions") is satisfied — the UI reads the same records the disk log gets.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def configure_root_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn's access log is noisy during SSE polling.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"cpl.{name}")


@dataclass
class JobEvent:
    """One thing that happened. `kind` drives how the UI renders it."""

    job_id: str
    kind: str  # stage | progress | log | solver | warning | error | state | done
    timestamp: float = field(default_factory=time.time)
    stage: str | None = None
    progress: float | None = None
    message: str = ""
    shot_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "message": self.message,
        }
        if self.stage is not None:
            d["stage"] = self.stage
        if self.progress is not None:
            d["progress"] = round(self.progress, 4)
        if self.shot_id is not None:
            d["shot_id"] = self.shot_id
        if self.data:
            d["data"] = self.data
        return d


class EventBus:
    """Thread-safe publish, asyncio-friendly subscribe.

    The pipeline runs in a worker thread; SSE handlers live on the event loop.
    Publishing therefore has to hop threads, which is why subscribers get an
    `asyncio.Queue` fed via `call_soon_threadsafe`.
    """

    def __init__(self, history: int = 500):
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = (
            defaultdict(list)
        )
        self._history: dict[str, deque[JobEvent]] = defaultdict(lambda: deque(maxlen=history))

    def publish(self, event: JobEvent) -> None:
        with self._lock:
            self._history[event.job_id].append(event)
            targets = list(self._subscribers.get(event.job_id, ()))
        for loop, queue in targets:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # Loop already closed (client vanished). Reaped on next unsubscribe.
                pass

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers[job_id].append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers[job_id] = [
                (l, q) for (l, q) in self._subscribers.get(job_id, ()) if q is not queue
            ]

    def replay(self, job_id: str) -> list[JobEvent]:
        """Events so far, so a client that connects mid-run sees the backlog."""
        with self._lock:
            return list(self._history.get(job_id, ()))

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._history.pop(job_id, None)


EVENT_BUS = EventBus()


class StageReporter:
    """Handed to each pipeline stage. Owns the job's log file and event stream.

    Progress is reported as a fraction of the *whole job*, computed from a stage
    weight table, so the UI progress bar does not jump backwards between stages.
    """

    #: Relative cost of each stage. Rough but stable; used only for the bar.
    STAGE_WEIGHTS = {
        "Preparing frames": 8,
        "Tracking features": 10,
        "Estimating 2D motion": 14,
        "Estimating camera geometry": 28,
        "Optimizing camera poses": 12,
        "Recovering lens motion": 4,
        "Validating trajectory": 4,
        "Creating Blender scene": 4,
        "Rendering MP4": 16,
    }

    def __init__(self, job_id: str, log_path: Any = None, bus: EventBus | None = None):
        self.job_id = job_id
        self.log_path = log_path
        self.bus = bus or EVENT_BUS
        self.log = get_logger(f"job.{job_id[:8]}")
        self._stage: str | None = None
        self._shot_id: int | None = None
        self._shot_index: int = 0
        self._shot_count: int = 1
        self._lock = threading.Lock()
        self._total_weight = sum(self.STAGE_WEIGHTS.values())
        self._last_overall: float = 0.0

    # -------------------------------------------------------------- internals

    def _write_disk(self, line: str) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line.rstrip() + "\n")
        except OSError:
            pass  # logging must never break the pipeline

    def _emit(self, event: JobEvent) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        prefix = f"[{stamp}] {event.kind.upper():8s}"
        if event.shot_id is not None:
            prefix += f" shot{event.shot_id:02d}"
        if event.stage:
            prefix += f" {event.stage}"
        self._write_disk(f"{prefix} :: {event.message}")
        self.bus.publish(event)

    def _overall_progress(self, stage: str, local: float) -> float:
        """Map (stage, local fraction) onto a monotonic global fraction.

        Monotonicity is enforced rather than assumed: stages can legitimately be
        skipped or revisited (a solver falling back down the ladder re-enters an
        earlier stage), and a progress bar that rewinds reads as a malfunction.
        """
        local = max(0.0, min(1.0, local))
        # Fold per-shot progress into this shot's slice of the stage.
        if self._shot_count > 1:
            local = (self._shot_index + local) / self._shot_count

        done = 0
        value = min(1.0, sum(self.STAGE_WEIGHTS.values()) / self._total_weight)
        for name, weight in self.STAGE_WEIGHTS.items():
            if name == stage:
                value = (done + weight * local) / self._total_weight
                break
            done += weight
        else:
            value = min(1.0, done / self._total_weight)

        with self._lock:
            if value < self._last_overall:
                value = self._last_overall
            else:
                self._last_overall = value
        return value

    # ----------------------------------------------------------------- public

    def set_shot(self, shot_id: int | None, *, index: int = 0, count: int = 1) -> None:
        """Scope subsequent progress to one shot of `count`.

        Without this a 3-shot job runs each per-shot stage from 0 to 100% three
        times and the bar visibly rewinds twice. `index`/`count` compress each
        shot's local progress into its own slice of the stage.
        """
        self._shot_id = shot_id
        self._shot_index = max(0, index)
        self._shot_count = max(1, count)

    def stage(self, stage: str, message: str = "") -> None:
        self._stage = stage
        self.log.info("stage=%s %s", stage, message)
        self._emit(
            JobEvent(
                self.job_id,
                "stage",
                stage=stage,
                progress=self._overall_progress(stage, 0.0),
                message=message or stage,
                shot_id=self._shot_id,
            )
        )

    def progress(self, fraction: float, message: str = "") -> None:
        stage = self._stage or ""
        self._emit(
            JobEvent(
                self.job_id,
                "progress",
                stage=stage,
                progress=self._overall_progress(stage, fraction),
                message=message,
                shot_id=self._shot_id,
                data={"stage_progress": round(max(0.0, min(1.0, fraction)), 4)},
            )
        )

    def info(self, message: str, **data: Any) -> None:
        self.log.info(message)
        self._emit(
            JobEvent(self.job_id, "log", stage=self._stage, message=message,
                     shot_id=self._shot_id, data=data)
        )

    def warning(self, message: str, **data: Any) -> None:
        self.log.warning(message)
        self._emit(
            JobEvent(self.job_id, "warning", stage=self._stage, message=message,
                     shot_id=self._shot_id, data=data)
        )

    def error(self, message: str, **data: Any) -> None:
        self.log.error(message)
        self._emit(
            JobEvent(self.job_id, "error", stage=self._stage, message=message,
                     shot_id=self._shot_id, data=data)
        )

    def solver_decision(self, decision: Any) -> None:
        """Record a SolverDecision. Invariant I9."""
        payload = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
        verdict = "selected" if payload.get("selected") else (
            "succeeded" if payload.get("succeeded") else "failed"
        )
        msg = f"{payload.get('solver')}: {verdict} — {payload.get('message', '')}".strip()
        self.log.info("solver decision %s", msg)
        self._emit(
            JobEvent(self.job_id, "solver", stage=self._stage, message=msg,
                     shot_id=self._shot_id, data=payload)
        )

    def state(self, state: str, message: str = "") -> None:
        self._emit(JobEvent(self.job_id, "state", message=message or state,
                            data={"state": state}))

    def done(self, message: str = "Complete") -> None:
        self._emit(JobEvent(self.job_id, "done", stage="Complete", progress=1.0, message=message))
