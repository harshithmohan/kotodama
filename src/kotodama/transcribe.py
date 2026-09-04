"""Stage 1: Japanese transcription via faster-whisper.

- Japanese (`language='ja'`) hardcoded — not configurable, no auto-detect
- condition_on_previous_text=False default (avoids repetition loops)
- vad_filter=True default (Silero VAD)
- task='transcribe' only; translation is stage 2's job.
- Falls back to CPU + int8 compute type when no GPU is available.
"""

from __future__ import annotations

import sys
from typing import Any

Segment = dict[str, Any]  # {"start": float, "end": float, "text": str}


def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' when a GPU is visible, else 'cpu'."""
    if requested and requested != "auto":
        return requested
    try:
        import ctranslate2  # bundled with faster-whisper

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe(
    media_path: str,
    model: str = "large-v3",
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    condition_on_previous_text: bool = False,
    device: str = "auto",
) -> list[Segment]:
    """Transcribe a media file and return a list of segments.

    Each segment is {"start": float_seconds, "end": float_seconds, "text": str}.
    Timestamps are never modified after this point.
    """
    from faster_whisper import WhisperModel  # heavy import, keep local

    resolved = _resolve_device(device)
    # int8 on CPU per spec; float16 on CUDA.
    compute_type = "int8" if resolved == "cpu" else "float16"

    print(
        f"[stage1] loading faster-whisper model '{model}' on {resolved} "
        f"(compute_type={compute_type})",
        file=sys.stderr,
    )
    whisper = WhisperModel(model, device=resolved, compute_type=compute_type)

    iterator, info = whisper.transcribe(
        media_path,
        language="ja",
        vad_filter=vad_filter,
        vad_parameters=vad_parameters or None,
        condition_on_previous_text=condition_on_previous_text,
        task="transcribe",
    )

    segments: list[Segment] = []
    for seg in iterator:
        segments.append(
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        )
        print(
            f"[stage1] {len(segments)} segments ({seg.end:6.1f}s / media)",
            file=sys.stderr,
            end="\r",
        )
    print(file=sys.stderr)
    if vad_filter and getattr(info, "duration_after_vad", None) is not None:
        print(
            f"[stage1] VAD kept {info.duration_after_vad:.1f}s of "
            f"{info.duration:.1f}s — if this is far below 100%, speech after "
            f"this point is being filtered; tune [transcribe.vad_parameters]",
            file=sys.stderr,
        )
    print(f"[stage1] done: {len(segments)} Japanese segments", file=sys.stderr)
    return segments
