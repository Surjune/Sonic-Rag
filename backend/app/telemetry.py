"""Per-stage latency measurement.

Every stage of a request is timed separately and reported to the caller. A
single end-to-end number hides which stage is actually slow, and in this
pipeline the stages differ by three orders of magnitude: FAISS runs in ~3ms
while a network hop to a model provider runs in hundreds.

Reporting each stage honestly is also what makes the interface's latency HUD
meaningful rather than decorative.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LatencyTrace:
    """Collects stage timings for one request."""

    stages: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it under `name`."""
        began = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - began) * 1000

    def record(self, name: str, milliseconds: float) -> None:
        """Record a stage measured elsewhere, e.g. inside a client."""
        self.stages[name] = milliseconds

    def mark(self, name: str, at: float | None = None) -> None:
        """Record elapsed time from the start of the request to a moment.

        Stages measure how long a step took. A mark measures when something
        became true for the user, which is a different question: the one that
        matters most here is when the first token of the answer appeared, since
        that is when reading can begin. Summing the stages would not give it --
        gaps between stages would vanish and any concurrency would double-count.
        """
        moment = at if at is not None else time.perf_counter()
        self.stages[name] = (moment - self._started) * 1000

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def as_dict(self) -> dict[str, float]:
        """Stage timings plus the true wall-clock total.

        The total is measured independently rather than summed: concurrent
        stages would otherwise double-count, and gaps between stages would
        vanish, both of which would flatter the numbers.
        """
        rounded = {name: round(value, 3) for name, value in self.stages.items()}
        rounded["total"] = round(self.total_ms, 3)
        return rounded
