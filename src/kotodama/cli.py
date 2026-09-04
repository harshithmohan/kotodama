"""CLI: kotodama INPUT [-o OUT.srt] [-c CONFIG.toml] [--endpoint URL] [--model NAME]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as toml_reader
except ModuleNotFoundError:  # pragma: no cover
    import tomli as toml_reader  # type: ignore[no-redef]

from . import __version__
from .subtitles import write_srt, write_translated_srt
from .translate import BackendError, OpenAICompatBackend
from .transcribe import transcribe

BUILTIN_DEFAULT_ENDPOINT = "http://localhost:11434"
ENV_ENDPOINT_VAR = "KOTODAMA_ENDPOINT"


def _load_config(path: str | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "transcribe": {
            "model": "large-v3",
            "vad_filter": True,
            "condition_on_previous_text": False,
            "device": "auto",
        },
        "translate": {
            # model "" = whatever model the endpoint loaded at startup
            "model": "",
            "endpoint": BUILTIN_DEFAULT_ENDPOINT,
            "context_window": 5,
        },
    }
    if path is None:
        # Auto-detect: cwd first, then the container's mounted location
        for candidate in (Path("config.toml"), Path("/app/config.toml")):
            if candidate.is_file():
                print(f"[config] using {candidate}", file=sys.stderr)
                path = str(candidate)
                break
    if path:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            _fail(f"config file not found: {cfg_path}", 2)
        with cfg_path.open("rb") as fh:
            loaded = toml_reader.load(fh)
        for section, values in loaded.items():
            if isinstance(values, dict):
                defaults.setdefault(section, {}).update(values)
            else:
                defaults[section] = values
    return defaults


def _fail(message: str, code: int) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def _derive_path(base: str, suffix: str) -> str:
    """Derive a sibling output path: output.english.srt -> output.<suffix>."""
    p = Path(base)
    if p.name.endswith(".english.srt"):
        stem = p.name[: -len(".english.srt")]
    else:
        stem = p.stem
    return str(p.with_name(f"{stem}.{suffix}"))


def _resolve_endpoint(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    """Resolution order: env KOTODAMA_ENDPOINT > CLI flag > config file > built-in default."""
    env = os.environ.get(ENV_ENDPOINT_VAR)
    if env:
        return env
    if args.endpoint:
        return args.endpoint
    return cfg["translate"].get("endpoint") or BUILTIN_DEFAULT_ENDPOINT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kotodama",
        description=(
            "Two-stage JA->EN subtitle pipeline: faster-whisper Japanese "
            "transcription, then LLM translation via an OpenAI-compatible "
            "endpoint."
        ),
    )
    parser.add_argument("input", help="input Japanese audio/video file")
    parser.add_argument(
        "-o",
        dest="out",
        default=None,
        help="output English SRT (default: <input>.srt next to the input file)",
    )
    parser.add_argument(
        "-c",
        dest="config",
        default=None,
        help="path to config.toml (default: auto-detect ./config.toml or /app/config.toml; falls back to built-in defaults)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(f"OpenAI-compatible endpoint URL (overridden by ${ENV_ENDPOINT_VAR})"),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model id served by the endpoint, or '' to use the endpoint's "
            "loaded model (default)"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        _fail(f"input file not found: {input_path}", 2)

    cfg = _load_config(args.config)
    tcfg, trcfg = cfg["transcribe"], cfg["translate"]

    model_name = args.model if args.model is not None else trcfg.get("model", "")
    endpoint = _resolve_endpoint(args, cfg)

    # Outputs default next to the input file (container-safe: /data is a mount).
    # English subs use the player-standard <input>.srt name so they auto-load.
    english_out = args.out or str(input_path.with_suffix(".srt"))
    ja_out = (
        _derive_path(english_out, "ja")
        if args.out
        else str(input_path.with_name(input_path.stem + ".ja.srt"))
    )

    # ---- Stage 1: Japanese transcription --------------------------------
    print(f"[pipeline] stage 1: transcribing '{input_path}' (JA)", file=sys.stderr)
    segments = transcribe(
        str(input_path),
        model=tcfg.get("model", "large-v3"),
        vad_filter=bool(tcfg.get("vad_filter", True)),
        vad_parameters=tcfg.get("vad_parameters") or None,
        condition_on_previous_text=bool(tcfg.get("condition_on_previous_text", False)),
        device=tcfg.get("device", "auto"),
    )
    if not segments:
        _fail("stage 1 produced no segments (no speech detected?)", 4)
    write_srt(segments, ja_out)

    # ---- Stage 2: JA -> EN translation (OpenAI-compatible endpoint) -----
    print(
        f"[pipeline] stage 2: translating {len(segments)} segments "
        f"JA->EN via OpenAI-compatible endpoint at {endpoint}",
        file=sys.stderr,
    )
    try:
        backend = OpenAICompatBackend(
            endpoint=endpoint,
            model=model_name,
            context_window=int(trcfg.get("context_window", 5)),
        )
    except BackendError as exc:
        _fail(str(exc), 3)

    texts = [seg["text"] for seg in segments]
    translations = backend.translate_batch(texts)

    if len(translations) != len(segments):
        _fail(
            f"backend returned {len(translations)} results for {len(segments)} "
            "segments (contract violation)",
            6,
        )
    failures = sum(1 for t in translations if not t)
    if failures:
        print(
            f"[pipeline] WARNING: {failures} segment(s) failed to translate "
            "(written as empty events)",
            file=sys.stderr,
        )
    write_translated_srt(segments, translations, english_out)

    print(
        f"[pipeline] done: {len(segments)} segments -> {english_out} "
        f"(Japanese: {ja_out})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
