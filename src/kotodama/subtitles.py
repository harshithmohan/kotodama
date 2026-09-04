"""Segment <-> SRT I/O via pysubs2.

Subtitle handling rules (OBJECTIVE.md): 1 segment in -> 1 event out,
timestamps never modified. Outputs are UTF-8 .srt.
"""

from __future__ import annotations

import sys
from typing import Any

import pysubs2

Segment = dict[str, Any]  # {"start": float, "end": float, "text": str}


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def write_srt(segments: list[Segment], path: str, text_key: str = "text") -> None:
    """Write segments to an SRT file. Timestamps are copied verbatim."""
    subs = pysubs2.SSAFile()
    for i, seg in enumerate(segments):
        text = seg.get(text_key)
        if text is None:
            text = ""
        start_ms, end_ms = _ms(seg["start"]), _ms(seg["end"])
        if start_ms == end_ms:
            # pysubs2's SRT writer drops zero-duration events; keep the 1-in/1-out
            # guarantee with the smallest representable duration (timestamps are
            # otherwise untouched).
            print(
                f"[note] segment {i + 1} rounds to zero duration; end nudged +1ms",
                file=sys.stderr,
            )
            end_ms += 1
        subs.append(
            pysubs2.SSAEvent(
                start=start_ms,  # int milliseconds (pysubs2 >= 1.7 API)
                end=end_ms,
                text=str(text),
            )
        )
    subs.save(path, format_="srt", encoding="utf-8")
    print(f"[output] wrote {len(segments)} events -> {path}", file=sys.stderr)


def write_translated_srt(
    segments: list[Segment], translations: list[str], path: str
) -> None:
    """Write an SRT where each event keeps segment i's timestamps but uses
    translations[i] as text. Lengths must match 1:1."""
    if len(translations) != len(segments):
        raise ValueError(
            f"translations ({len(translations)}) != segments ({len(segments)}); "
            "refusing to write mismatched subtitles"
        )
    merged = [
        {"start": seg["start"], "end": seg["end"], "text": tr}
        for seg, tr in zip(segments, translations)
    ]
    _flag_timing_outliers(merged)
    write_srt(merged, path)


def _flag_timing_outliers(segments: list[Segment]) -> None:
    """v1 does not retime; only log extreme length/duration mismatches."""
    for i, seg in enumerate(segments, 1):
        duration = max(seg["end"] - seg["start"], 0.01)
        cps = len(seg["text"]) / duration
        if cps > 20.0 or cps < 0.5:
            print(
                f"[note] segment {i}: text length vs time slot is extreme "
                f"({cps:.1f} chars/s); retiming is out of scope for v1",
                file=sys.stderr,
            )


def read_srt(path: str) -> list[Segment]:
    """Read an SRT back into segment dicts (start/end in seconds)."""
    subs = pysubs2.load(path, format_="srt", encoding="utf-8")
    return [
        {
            "start": event.start / 1000.0,
            "end": event.end / 1000.0,
            "text": str(event.text).strip(),
        }
        for event in subs
    ]
