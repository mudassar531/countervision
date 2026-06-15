"""Unit tests for ``countervision.timeparse``."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Allow ``import countervision`` without installing the package.
_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.timeparse import (  # noqa: E402
    RecordingStart,
    TimeparseError,
    parse_recording_start_from_name,
    resolve_recording_start,
    wall_clock_for_frame,
)


class TestParseRecordingStartFromName:
    def test_plain_mp4(self) -> None:
        assert parse_recording_start_from_name("20260607205350587.mp4") == datetime(
            2026, 6, 7, 20, 53, 50, 587_000
        )

    def test_suffixed_mov(self) -> None:
        # Real camera-5 filename pattern from this project.
        assert parse_recording_start_from_name(
            "20260608044448561_AH8174419_Barkerend - Kingz_16_video.mov"
        ) == datetime(2026, 6, 8, 4, 44, 48, 561_000)

    def test_just_stem(self) -> None:
        assert parse_recording_start_from_name("20260608005449323") == datetime(
            2026, 6, 8, 0, 54, 49, 323_000
        )

    def test_full_path(self) -> None:
        assert parse_recording_start_from_name(
            "/videos/camera-3/20260608003129784.mp4"
        ) == datetime(2026, 6, 8, 0, 31, 29, 784_000)

    @pytest.mark.parametrize(
        "name",
        [
            "weird_name.mp4",
            "2026060720535058.mp4",   # 16 digits
            "202606072053505876.mp4",  # 18 digits
            "abcdefghijklmnopq.mp4",
        ],
    )
    def test_unparseable_returns_none(self, name: str) -> None:
        assert parse_recording_start_from_name(name) is None

    def test_invalid_calendar_date_returns_none(self) -> None:
        # Month 13 → invalid calendar date → None (not an exception).
        assert parse_recording_start_from_name("20261307205350587.mp4") is None


class TestResolveRecordingStart:
    def test_filename_wins_even_when_file_missing(self) -> None:
        result = resolve_recording_start("/no/such/file/20260607205350587.mp4")
        assert result.source == "filename"
        assert result.is_from_filename
        assert result.timestamp == datetime(2026, 6, 7, 20, 53, 50, 587_000)

    def test_mtime_fallback_when_filename_unparseable(self, tmp_path: Path) -> None:
        p = tmp_path / "weird_name.mp4"
        p.write_bytes(b"")
        result = resolve_recording_start(p)
        assert result.source == "mtime"
        assert not result.is_from_filename

    def test_raises_when_unparseable_and_missing(self) -> None:
        with pytest.raises(TimeparseError):
            resolve_recording_start("/no/such/file/weird_name.mp4")

    def test_isoformat_includes_milliseconds(self) -> None:
        rs = RecordingStart(
            timestamp=datetime(2026, 6, 7, 20, 53, 50, 587_000),
            source="filename",
        )
        assert rs.isoformat() == "2026-06-07T20:53:50.587"


class TestWallClockForFrame:
    def test_arithmetic(self) -> None:
        rs = RecordingStart(
            timestamp=datetime(2026, 6, 7, 20, 53, 50, 587_000),
            source="filename",
        )
        # 25 fps, frame 250 → +10.0s
        assert wall_clock_for_frame(rs, 250, 25.0) == datetime(
            2026, 6, 7, 20, 54, 0, 587_000
        )

    def test_zero_fps_raises(self) -> None:
        rs = RecordingStart(
            timestamp=datetime(2026, 6, 7, 0, 0, 0),
            source="filename",
        )
        with pytest.raises(ValueError):
            wall_clock_for_frame(rs, 1, 0.0)
