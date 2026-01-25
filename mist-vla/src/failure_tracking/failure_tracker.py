"""
Lightweight failure tracking scaffold.
Not connected to the rollout pipeline yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FailureEvent:
    step: int
    reason: str
    risk_vector: Optional[np.ndarray] = None


@dataclass
class FailureTracker:
    """
    Collects failure-related events and signals for later analysis.
    """

    events: List[FailureEvent] = field(default_factory=list)

    def record(self, step: int, reason: str, risk_vector: Optional[np.ndarray] = None) -> None:
        self.events.append(FailureEvent(step=step, reason=reason, risk_vector=risk_vector))

    def summarize(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for event in self.events:
            summary[event.reason] = summary.get(event.reason, 0) + 1
        return summary

    def reset(self) -> None:
        self.events = []
