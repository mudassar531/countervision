"""Click-to-draw zone editor + non-interactive default-zone generator.

Two ways to populate zones / entry lines for a camera:

* ``--draw-zones CAMERA_ID`` — opens an OpenCV window with the saved
  first frame; left-click to add polygon vertices, ``c`` to close the
  current polygon, ``n`` to start a new polygon, ``l`` to toggle line
  mode (two clicks: start, then end), ``s`` to save, ``q`` to quit.
  Writes the camera block back into ``config.yaml``.

* ``--draw-zones-default`` — no GUI. Generates one sensible "main floor"
  polygon (central 60 % of the frame) and one horizontal entry line at
  75 % of frame height for every camera in the config. Idempotent; only
  writes a camera block that has empty zones / entry_line, so manual
  edits are preserved.

Either way the config schema is the same so the rest of the pipeline
doesn't care which path produced the coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)


def default_zones_for_frame(width: int, height: int) -> list[dict[str, Any]]:
    """One central "main floor" polygon covering ~60 % of the frame area."""
    margin_x = int(round(width * 0.20))
    margin_y = int(round(height * 0.20))
    polygon = [
        [margin_x, margin_y],
        [width - margin_x, margin_y],
        [width - margin_x, height - margin_y],
        [margin_x, height - margin_y],
    ]
    return [{"name": "Main floor", "color": "#0A1347", "polygon": polygon}]


def default_entry_line_for_frame(width: int, height: int) -> dict[str, Any]:
    """Horizontal line at 75 % of frame height; left → right = "in"."""
    y = int(round(height * 0.75))
    margin_x = int(round(width * 0.15))
    return {"start": [margin_x, y], "end": [width - margin_x, y]}


def load_config_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_config_yaml(config_path: Path, data: dict[str, Any]) -> None:
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=None, width=120)


def populate_defaults(
    *,
    config_path: Path,
    frames_dir: Path,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Populate config.yaml with default zones + entry lines per camera.

    Reads ``frames/<camera>.jpg`` to get the frame size per camera. If a
    camera already has non-empty ``zones`` / ``entry_line`` and
    ``overwrite=False`` (the default), it's left alone.

    To avoid PyYAML stripping comments from unrelated blocks on every
    invocation, we only write the file back when at least one camera
    block actually changed.
    """
    cfg = load_config_yaml(config_path)
    cameras = cfg.get("cameras") or {}
    summary: dict[str, dict[str, Any]] = {}
    file_dirty = False

    for cam_id, block in cameras.items():
        frame_path = frames_dir / f"{cam_id}.jpg"
        if not frame_path.exists():
            summary[cam_id] = {"skipped": "no frame jpg yet — run --run-detect-track first"}
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            summary[cam_id] = {"skipped": f"could not read {frame_path}"}
            continue
        h, w = img.shape[:2]
        block = dict(block or {})
        cam_changes: dict[str, Any] = {"frame_size": [w, h]}

        if overwrite or not block.get("zones"):
            block["zones"] = default_zones_for_frame(w, h)
            cam_changes["zones"] = "default (central 60% of frame)"
            file_dirty = True
        else:
            cam_changes["zones"] = f"kept ({len(block['zones'])} existing)"

        if overwrite or block.get("entry_line") in (None, [], {}):
            block["entry_line"] = default_entry_line_for_frame(w, h)
            cam_changes["entry_line"] = "default (horizontal at 75% height)"
            file_dirty = True
        else:
            cam_changes["entry_line"] = "kept (existing)"

        cameras[cam_id] = block
        summary[cam_id] = cam_changes

    if file_dirty:
        cfg["cameras"] = cameras
        write_config_yaml(config_path, cfg)
    return summary


@dataclass
class _EditorState:
    width: int
    height: int
    polygons: list[dict[str, Any]] = field(default_factory=list)
    current_polygon: list[list[int]] = field(default_factory=list)
    entry_line: list[list[int]] = field(default_factory=list)
    mode: str = "polygon"

    def add_point(self, x: int, y: int) -> None:
        if self.mode == "polygon":
            self.current_polygon.append([x, y])
        else:
            self.entry_line.append([x, y])
            if len(self.entry_line) > 2:
                self.entry_line = self.entry_line[-2:]

    def close_polygon(self, name: str) -> None:
        if len(self.current_polygon) >= 3:
            self.polygons.append(
                {
                    "name": name,
                    "color": "#0A1347",
                    "polygon": [list(map(int, p)) for p in self.current_polygon],
                }
            )
            self.current_polygon = []


def _draw_state(canvas: np.ndarray, st: _EditorState) -> np.ndarray:
    img = canvas.copy()
    for poly in st.polygons:
        pts = np.array(poly["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.putText(
            img, poly["name"], tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    if st.current_polygon:
        pts = np.array(st.current_polygon, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 200, 255), thickness=2)
        for p in st.current_polygon:
            cv2.circle(img, tuple(p), 4, (0, 200, 255), -1)
    if len(st.entry_line) == 2:
        cv2.arrowedLine(
            img,
            tuple(st.entry_line[0]),
            tuple(st.entry_line[1]),
            (0, 255, 0), 2, tipLength=0.05,
        )
    hud_lines = [
        f"mode: {st.mode}",
        "l: line  n: new polygon  c: close polygon  s: save  q: quit",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(
            img, line, (16, 28 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (16, 28 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def interactive_draw(  # pragma: no cover — GUI-dependent
    *,
    camera_id: str,
    frame_path: Path,
    config_path: Path,
    zone_names: list[str] | None = None,
) -> dict[str, Any]:
    """Open a cv2 window and let the operator click to draw zones + line.

    Returns the camera block written back to ``config.yaml``.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"could not read {frame_path}")
    height, width = img.shape[:2]
    state = _EditorState(width=width, height=height)
    names_iter = iter(zone_names or [f"Zone {i + 1}" for i in range(10)])
    current_name = next(names_iter, "Zone 1")

    def _on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_point(x, y)

    win = f"draw zones [{camera_id}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)

    while True:
        cv2.imshow(win, _draw_state(img, state))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            return {}
        if key == ord("n"):
            state.close_polygon(current_name)
            current_name = next(names_iter, f"Zone {len(state.polygons) + 1}")
        elif key == ord("c"):
            state.close_polygon(current_name)
        elif key == ord("l"):
            state.mode = "line" if state.mode == "polygon" else "polygon"
        elif key == ord("s"):
            state.close_polygon(current_name)
            cfg = load_config_yaml(config_path)
            cameras = cfg.setdefault("cameras", {})
            cam_block = dict(cameras.get(camera_id) or {})
            cam_block["zones"] = state.polygons
            if len(state.entry_line) == 2:
                cam_block["entry_line"] = {
                    "start": list(map(int, state.entry_line[0])),
                    "end": list(map(int, state.entry_line[1])),
                }
            cameras[camera_id] = cam_block
            cfg["cameras"] = cameras
            write_config_yaml(config_path, cfg)
            cv2.destroyWindow(win)
            return cam_block
