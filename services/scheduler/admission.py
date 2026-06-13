"""Admission control & backpressure (DESIGN.md §5.2).

"Unbounded queues are how systems die; bound them explicitly." The gateway calls
this before enqueue. The decision is a pure function of the current depth and the
configured watermarks so it is testable and identical in the live path and any
simulation:

  * depth ≥ high watermark           → SHED (gateway returns 429),
  * low ≤ depth < high               → ACCEPT but signal "degraded" (clients may
                                        back off; HPA is already scaling on depth),
  * depth < low                      → ACCEPT.

The watermarks are expressed in queued items. The autoscaler reacts to the same
``relay_queue_depth`` gauge (DESIGN.md ADR-9), so admission control and scaling
are driven by one consistent signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Admission(StrEnum):
    ACCEPT = "accept"
    DEGRADED = "degraded"  # accepted, but over the soft limit
    SHED = "shed"  # rejected (429)


@dataclass(slots=True)
class Watermarks:
    low: int = 2_000
    high: int = 10_000

    def __post_init__(self) -> None:
        if self.low < 0 or self.high <= self.low:
            raise ValueError("require 0 <= low < high")


def classify(depth: int, marks: Watermarks) -> Admission:
    if depth >= marks.high:
        return Admission.SHED
    if depth >= marks.low:
        return Admission.DEGRADED
    return Admission.ACCEPT


def should_admit(depth: int, marks: Watermarks) -> bool:
    """True unless we are at/over the hard high-water mark."""
    return classify(depth, marks) is not Admission.SHED
