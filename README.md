# kotodama

Generate **English subtitles (SRT)** from **Japanese audio/video** using a
two-stage pipeline whose output is noticeably better than Whisper's built-in
`task='translate'` (which was evaluated and rejected as unacceptable — it runs
ASR and translation in one model and produces rough, unnatural English).

LLMs cannot replace the ASR stage (they take text, not audio), so the design
splits the problem: keep a dedicated speech recognizer for accurate Japanese
transcription and hand *only the text* to a capable LLM for fluent translation.

## Architecture

```
media file
   │
   ▼
[Stage 1: Transcribe (JA)]   faster-whisper (large-v3 default)
   │  → segments {start, end, text}   ← timestamps originate here and are NEVER modified
   ▼
[Stage 2: Translate (JA→EN)] OpenAI-compatible endpoint
   │  → translated text per segment (context-window batching, 1:1 out)
   ▼
SRT writers (pysubs2)
```

### Stage 1 — transcription (faster-whisper)

- Model `large-v3` default (configurable, including a local CTranslate2 path);
  `language='ja'` fixed — no auto-detection.
  - **Why not kotoba-whisper?** It transcribes Japanese TV audio slightly better
    and is ~6x faster, but its training data (ReazonSpeech 5s shards) carries no
    timestamp annotation, so its timestamps are unreliable in faster-whisper
    (the official stable-ts fix only exists in the transformers `v2.1`
    pipeline). Timestamp fidelity is critical for subtitles, so `large-v3`
    stays the default; a two-pass kotoba+aligner approach is a future option.
- `vad_filter=True` (Silero) to skip non-speech; `condition_on_previous_text=False`
  to avoid repetition loops on long-form Japanese.
- **`task='transcribe'` only** — built-in translation is never used; translation
  is stage 2's job.
- Device `auto`; CPU falls back to `int8` compute type.

### Stage 2 — translation (OpenAI-compatible endpoint)

- Implements `translate_batch(texts) -> list[str]`, strictly 1:1 with input
  (same length; empty string + logged warning on per-segment failure).
- Requests batch ~10 segments at a time with `context_window` surrounding
  segments as context so the model translates consistently; on response
  mismatch the batch falls back to per-segment requests.
- Endpoint resolution order: **`KOTODAMA_ENDPOINT` env > `--endpoint` flag >
  config file > built-in default `http://localhost:11434`**.

### Subtitle handling rules

- 1 segment in → 1 subtitle event out; timestamps untouched (zero-duration
  events get +1 ms with a logged note, because pysubs2 silently drops them).
- No re-timing or line-wrapping to CPS constraints in v1 — length-vs-time-slot
  outliers are noted in the log.
- Outputs, written next to the input file: `<input>.srt` (pipeline translation,
  named so players auto-load it) and `<input>.ja.srt` (raw Japanese transcript,
  for QA and re-runs).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+. CPU-only works (faster-whisper falls back to `int8`);
a GPU makes `large-v3` much faster. Dependencies: `faster-whisper`, `pysubs2`,
`requests`.

## Configure

```bash
cp config.example.toml config.toml
```

Translation runs against an **OpenAI-compatible chat endpoint** (e.g.
`http://atlantis:11435`, serving **`gemma4-26b-a4b_q4`**). The endpoint serves
a **single model loaded at server startup**, so `[translate].model` may be left
`""` to use whatever is loaded; set it to a model id to have it verified
against `GET /v1/models` (and sent as the `model` field).

- Note: `gemma4-26b-a4b_q4` is a **reasoning model** — chain-of-thought
  arrives in a separate `reasoning_content` field which the backend ignores,
  so parsing is correct, but batched requests are slower than with a
  non-reasoning model.
- **Server context size (`-c` / `n_ctx`): 8192 recommended** (4096 minimum if
  VRAM-bound; ≥16k is wasted — a request carries ~1–2k input tokens).

Env override example:

```bash
export KOTODAMA_ENDPOINT="http://atlantis:11435"
```

## Usage

```bash
# basic: writes video.srt + video.ja.srt next to the input
kotodama video.mp4

# explicit config and output path (Japanese transcript is derived alongside: out.ja.srt)
kotodama episode.mkv -c config.toml -o out/english.srt

# endpoint / model overrides
kotodama video.mp4 --endpoint http://atlantis:11435
kotodama video.mp4 --endpoint http://atlantis:11435 \
    --model gemma4-26b-a4b_q4   # or --model "" to use the server's loaded model
```

`--help` lists all flags. Progress is printed to stderr; exit codes are
non-zero with a clear message on missing input (2), unreachable endpoint /
missing model (3), no speech detected (4), or a segment-count violation (6).

## Project layout

```
src/kotodama/
├── transcribe.py   # stage 1 (faster-whisper)
├── translate.py    # stage 2 (OpenAICompatBackend, /v1/chat/completions)
├── subtitles.py    # segment <-> SRT io (pysubs2)
└── cli.py          # kotodama entry point
config.example.toml # template (copy to config.toml on the host / mount point)
tests/test_smoke.py # offline smoke tests (stub HTTP server, SRT round-trip)
```

## Docker (CUDA)

Base image: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (cuDNN 9 preinstalled;
CTranslate2 is tested against CUDA 12.x — newer 13.x images work with driver
≥ 525 but are not what faster-whisper targets). Requires
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host; any driver ≥ 525 works (tested against 595.91.07).

```bash
docker build -t kotodama .

# first run downloads large-v3 into the mounted /models volume (persisted)
docker run --rm --gpus all \
  -v /path/to/media:/data \
  -v /path/to/models:/models \
  kotodama /data/video.mp4 -c /data/config.toml
```

`device: "auto"` (default) detects the GPU and uses `float16`; on a plain CPU
container it falls back to `int8` with no config change.

## Known limitations (v1)

- No speaker diarization.
- No subtitle re-timing / CPS-aware line wrapping.
- Japanese→English only.
- Reasoning models are slower per request (reasoning tokens are
  generated but discarded).

## Tests

Offline smoke tests (SRT round-trip + the OpenAI-compatible backend against a stub HTTP
server — no real endpoint or media needed):

```bash
python tests/test_smoke.py    # or: pytest tests/
```
