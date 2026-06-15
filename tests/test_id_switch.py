"""Unit tests for the IdSwitchCounter spatial-overlap proxy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.id_switch import IdSwitchCounter, iou_xyxy  # noqa: E402


def _box(x: float, y: float, w: float = 40, h: float = 80) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=float)


def test_iou_disjoint_is_zero() -> None:
    assert iou_xyxy(_box(0, 0), _box(500, 500)) == 0.0


def test_iou_identical_is_one() -> None:
    assert iou_xyxy(_box(10, 10), _box(10, 10)) == 1.0


def test_no_switch_when_id_persists() -> None:
    counter = IdSwitchCounter(iou_threshold=0.3, lookback_frames=30)
    box = _box(100, 100)
    # Same id, slight drift across frames → no switch.
    for frame in range(5):
        counter.update(frame, np.array([1]), np.array([box + np.array([frame, frame, frame, frame])]))
    assert counter.count == 0
    assert counter.seen_ids == {1}


def test_one_switch_when_id_changes_in_same_spot() -> None:
    counter = IdSwitchCounter(iou_threshold=0.3, lookback_frames=30)
    box = _box(200, 200)
    counter.update(0, np.array([1]), np.array([box]))
    counter.update(1, np.array([1]), np.array([box]))
    # frame 2: tid 1 disappears; brand-new tid 7 appears at the same spot.
    counter.update(2, np.array([7]), np.array([box.copy()]))
    assert counter.count == 1
    assert counter.switches[0]["lost_id"] == 1
    assert counter.switches[0]["new_id"] == 7
    assert counter.switches[0]["iou"] >= 0.99


def test_no_switch_when_new_id_appears_in_different_spot() -> None:
    counter = IdSwitchCounter(iou_threshold=0.3, lookback_frames=30)
    counter.update(0, np.array([1]), np.array([_box(100, 100)]))
    # A genuinely new person appearing elsewhere should not count.
    counter.update(1, np.array([2]), np.array([_box(800, 600)]))
    assert counter.count == 0


def test_no_switch_when_lookback_exceeded() -> None:
    counter = IdSwitchCounter(iou_threshold=0.3, lookback_frames=10)
    box = _box(300, 300)
    counter.update(0, np.array([1]), np.array([box]))
    # New tid in the same spot but 50 frames later → past lookback, ignore.
    counter.update(50, np.array([42]), np.array([box.copy()]))
    assert counter.count == 0


def test_switch_picks_best_iou_partner() -> None:
    counter = IdSwitchCounter(iou_threshold=0.3, lookback_frames=30)
    # Two prior tracks; the new id is closer to id 5 than to id 2.
    counter.update(0, np.array([2]), np.array([_box(100, 100)]))
    counter.update(0, np.array([5]), np.array([_box(400, 100)]))
    counter.update(1, np.array([2]), np.array([_box(101, 100)]))
    counter.update(1, np.array([5]), np.array([_box(401, 100)]))
    counter.update(2, np.array([99]), np.array([_box(400, 100)]))  # close to id 5
    assert counter.count == 1
    assert counter.switches[0]["lost_id"] == 5
    assert counter.switches[0]["new_id"] == 99
