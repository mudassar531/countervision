"""Spatial-overlap proxy for ID switches.

Lifted out of ``detect_track.py`` so the unit tests can exercise it
without pulling in torch / ultralytics / supervision — those only matter
when actually running the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two ``[x1,y1,x2,y2]`` boxes (zero if disjoint)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class IdSwitchCounter:
    """Spatial-overlap proxy for ID switches when no ground truth is available.

    Whenever a previously-unseen ``tracker_id`` appears, check if any other
    ``tracker_id`` was seen in the last ``lookback_frames`` frames whose last
    bounding box overlaps the new one with IoU ≥ ``iou_threshold``. If yes,
    record one likely switch (lost_id → new_id) at the current frame.

    This is a *churn* metric, not MOTA — there is no ground-truth set of
    person tracks to compare against.
    """

    iou_threshold: float = 0.30
    lookback_frames: int = 30

    last_seen: dict[int, tuple[np.ndarray, int]] = field(default_factory=dict)
    seen_ids: set[int] = field(default_factory=set)
    switches: list[dict[str, Any]] = field(default_factory=list)

    def update(self, frame_idx: int, ids: np.ndarray, boxes: np.ndarray) -> None:
        for tid_raw, box in zip(ids, boxes, strict=True):
            tid = int(tid_raw)
            if tid not in self.seen_ids:
                self.seen_ids.add(tid)
                best: tuple[int, float] | None = None
                for prev_id, (prev_box, prev_frame) in self.last_seen.items():
                    if prev_id == tid:
                        continue
                    if frame_idx - prev_frame > self.lookback_frames:
                        continue
                    iou = iou_xyxy(prev_box, box)
                    if iou >= self.iou_threshold and (best is None or iou > best[1]):
                        best = (prev_id, iou)
                if best is not None:
                    self.switches.append(
                        {
                            "frame_idx": frame_idx,
                            "lost_id": best[0],
                            "new_id": tid,
                            "iou": round(best[1], 4),
                        }
                    )
            self.last_seen[tid] = (box.copy(), frame_idx)

    @property
    def count(self) -> int:
        return len(self.switches)
