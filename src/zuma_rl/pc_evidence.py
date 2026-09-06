"""Strict parsers for PC golden sidecar evidence.

Artifact hashes only bind bytes; they do not prove that a sidecar describes
the manifest.  This module parses the v1 per-tick PTS table and cross-checks it
against the video range and canonical trace.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from zuma_rl.pc_golden import (
    PcGoldenManifest,
    PcGoldenTrace,
    PcGoldenValidationError,
    TickClock,
    VideoMetadata,
)

_ROW_PATTERN = re.compile(r"(0|[1-9][0-9]*),(-?(?:0|[1-9][0-9]*))")


@dataclass(frozen=True, slots=True)
class TickPts:
    """Representative presentation timestamp for one native tick."""

    tick: int
    pts: int

    def __post_init__(self) -> None:
        if isinstance(self.tick, bool) or not isinstance(self.tick, int):
            raise PcGoldenValidationError("tick-map tick must be an integer")
        if self.tick < 0:
            raise PcGoldenValidationError(
                "tick-map tick must be non-negative"
            )
        if isinstance(self.pts, bool) or not isinstance(self.pts, int):
            raise PcGoldenValidationError("tick-map PTS must be an integer")


@dataclass(frozen=True, slots=True)
class PcTickMap:
    """Canonical ``tick,pts`` table for every native 100 Hz tick."""

    records: tuple[TickPts, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise PcGoldenValidationError("tick-map must not be empty")
        previous_pts: int | None = None
        for expected_tick, record in enumerate(records):
            if not isinstance(record, TickPts):
                raise PcGoldenValidationError(
                    "tick-map records must contain only TickPts objects"
                )
            if record.tick != expected_tick:
                raise PcGoldenValidationError(
                    "tick-map ticks must be contiguous and start at zero; "
                    f"expected {expected_tick}, got {record.tick}"
                )
            if previous_pts is not None and record.pts <= previous_pts:
                raise PcGoldenValidationError(
                    "tick-map PTS values must be strictly increasing"
                )
            previous_pts = record.pts
        object.__setattr__(self, "records", records)

    def to_csv(self) -> str:
        return "tick,pts\n" + "".join(
            f"{record.tick},{record.pts}\n" for record in self.records
        )

    @classmethod
    def from_csv(cls, text: str) -> "PcTickMap":
        if not isinstance(text, str):
            raise TypeError("tick-map text must be a string")
        lines = text.splitlines()
        if not lines or lines[0] != "tick,pts":
            raise PcGoldenValidationError(
                "tick-map must start with the exact header 'tick,pts'"
            )
        records: list[TickPts] = []
        for line_number, line in enumerate(lines[1:], start=2):
            match = _ROW_PATTERN.fullmatch(line)
            if match is None or match.group(2) == "-0":
                raise PcGoldenValidationError(
                    f"tick-map line {line_number} is not canonical"
                )
            try:
                tick = int(match.group(1))
                pts = int(match.group(2))
            except ValueError as error:
                raise PcGoldenValidationError(
                    f"tick-map line {line_number} contains an integer "
                    "outside the parser limit"
                ) from error
            records.append(TickPts(tick=tick, pts=pts))
        result = cls(records=tuple(records))
        if text != result.to_csv():
            raise PcGoldenValidationError(
                "tick-map CSV is not in canonical v1 form"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcTickMap":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "tick-map could not be read as UTF-8"
            ) from error
        return cls.from_csv(text)

    def validate_against(
        self,
        manifest: PcGoldenManifest,
        trace: PcGoldenTrace,
    ) -> None:
        """Cross-check PTS rows, frame references, and input containment."""

        self.validate_against_run(
            manifest,
            trace,
            tick_map_artifact=manifest.clock.tick_map_artifact,
            video=manifest.video,
            clock=manifest.clock,
        )

    def validate_against_run(
        self,
        manifest: PcGoldenManifest,
        trace: PcGoldenTrace,
        *,
        tick_map_artifact: str,
        video: VideoMetadata,
        clock: TickClock,
    ) -> None:
        """Cross-check one primary or replay tick-map against raw evidence."""

        if not isinstance(manifest, PcGoldenManifest):
            raise TypeError("manifest must be a PcGoldenManifest")
        if not isinstance(trace, PcGoldenTrace):
            raise TypeError("trace must be a PcGoldenTrace")
        if not isinstance(video, VideoMetadata):
            raise TypeError("video must be VideoMetadata")
        if not isinstance(clock, TickClock):
            raise TypeError("clock must be TickClock")
        if clock.mapping_kind != "per_tick_pts_table":
            raise PcGoldenValidationError(
                "PC golden requires mapping_kind='per_tick_pts_table'"
            )
        if tick_map_artifact != clock.tick_map_artifact:
            raise PcGoldenValidationError(
                "tick-map artifact does not match the run clock"
            )
        try:
            artifact = manifest.artifacts[tick_map_artifact]
        except KeyError:
            raise PcGoldenValidationError(
                "tick-map artifact is not present in the manifest"
            ) from None
        payload = self.to_csv().encode("utf-8")
        payload_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if (
            len(payload) != artifact.bytes
            or payload_sha256 != artifact.sha256
        ):
            raise PcGoldenValidationError(
                "canonical tick-map identity does not match the manifest"
            )
        expected_count = clock.tick_end + 1
        if len(self.records) != expected_count:
            raise PcGoldenValidationError(
                "tick-map row count does not match clock.tick_end"
            )
        if self.records[0].pts != clock.tick0_video_pts:
            raise PcGoldenValidationError(
                "tick-map tick-zero PTS does not match the manifest"
            )

        for record, golden_tick in zip(
            self.records,
            trace.records,
            strict=True,
        ):
            if not (
                video.first_pts
                <= record.pts
                <= video.last_pts
            ):
                raise PcGoldenValidationError(
                    f"tick-map PTS is outside video at tick {record.tick}"
                )
            if record.pts not in {frame.pts for frame in golden_tick.frames}:
                raise PcGoldenValidationError(
                    "trace does not reference the tick-map presentation PTS "
                    f"at tick {record.tick}"
                )
            if record.tick == 0:
                continue
            previous_pts = self.records[record.tick - 1].pts
            for item in golden_tick.inputs:
                if not previous_pts < item.pts <= record.pts:
                    raise PcGoldenValidationError(
                        "input PTS is outside its containing native-tick "
                        f"interval at tick {record.tick}"
                    )


__all__ = ["PcTickMap", "TickPts"]
