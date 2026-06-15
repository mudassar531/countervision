"""Parse recording start-time from CCTV filenames.

The cameras encode the recording start as the leading
``YYYYMMDDHHMMSSmmm`` (17 digits) of the filename, optionally followed by
``_...`` (free-text suffix from the NVR). The burned-in clock in some files
disagrees with the filename — **the filename is treated as authoritative**, per
the build spec.

If a filename can't be parsed we fall back to the file's mtime and flag the
result so callers can surface "(from mtime)" in their output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# 17 leading digits: YYYY MM DD HH MM SS mmm
_FILENAME_RE = re.compile(r"^(?P<ts>\d{17})(?:[_.\-].*)?$")


class TimeparseError(ValueError):
    """Raised when both filename parsing and mtime fallback fail."""


@dataclass(frozen=True)
class RecordingStart:
    """Result of resolving a video file's recording start time."""

    timestamp: datetime
    source: str  # "filename" or "mtime"

    @property
    def is_from_filename(self) -> bool:
        return self.source == "filename"

    def isoformat(self) -> str:
        return self.timestamp.isoformat(timespec="milliseconds")


def parse_recording_start_from_name(name: str) -> datetime | None:
    """Parse ``YYYYMMDDHHMMSSmmm`` from a filename stem; return ``None`` if absent.

    Accepts either the full filename or just the stem (extension is stripped).
    """
    stem = Path(name).stem
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    ts = m.group("ts")
    try:
        return datetime(
            year=int(ts[0:4]),
            month=int(ts[4:6]),
            day=int(ts[6:8]),
            hour=int(ts[8:10]),
            minute=int(ts[10:12]),
            second=int(ts[12:14]),
            microsecond=int(ts[14:17]) * 1000,
        )
    except ValueError as exc:
        log.warning("Filename %r had 17 digits but is not a valid timestamp: %s", name, exc)
        return None


def resolve_recording_start(path: str | Path) -> RecordingStart:
    """Resolve a video file's recording start time.

    Tries the filename first (authoritative); falls back to ``mtime`` and logs
    a warning. Raises :class:`TimeparseError` if the file doesn't exist and
    no filename timestamp can be parsed.
    """
    p = Path(path)
    parsed = parse_recording_start_from_name(p.name)
    if parsed is not None:
        return RecordingStart(timestamp=parsed, source="filename")

    if not p.exists():
        raise TimeparseError(
            f"Cannot resolve recording start for {p}: filename does not match "
            "YYYYMMDDHHMMSSmmm and the file does not exist for mtime fallback."
        )

    log.warning(
        "Filename %r is not parseable as YYYYMMDDHHMMSSmmm; falling back to mtime.",
        p.name,
    )
    return RecordingStart(
        timestamp=datetime.fromtimestamp(p.stat().st_mtime),
        source="mtime",
    )


def wall_clock_for_frame(start: RecordingStart, frame_index: int, fps: float) -> datetime:
    """Return the wall-clock timestamp of a given frame index."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return start.timestamp + timedelta(seconds=frame_index / fps)
