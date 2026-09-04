"""Offline smoke tests for kotodama (no GPU, no real translation endpoint needed).

Run: python tests/test_smoke.py   (also works under pytest, but needs no deps
beyond the project itself + stdlib).

(a) subtitles.py round-trip: fake segments -> .srt -> re-read -> compare.
(b) OpenAI-compatible backend against a stub HTTP server: canned /v1/models and
    /v1/chat/completions, verifying translate_batch is 1:1, the per-segment
    fallback works, empty-model skips verification, and unreachable endpoint /
    missing model produce clear actionable errors.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kotodama.subtitles import read_srt, write_srt, write_translated_srt
from kotodama.translate import (
    BackendError,
    OpenAICompatBackend,
    ModelNotFoundError,
)

FAKE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "こんにちは、今日はいい天気ですね。"},
    {"start": 2.5, "end": 5.0, "text": "ええ、散歩には最適です。"},
    {"start": 5.123, "end": 8.999, "text": "あの店のカフェ、try しましたか？"},
]


# ---------------------------------------------------------------------------
# Stub OpenAI-compatible server (/v1/models + /v1/chat/completions)
# ---------------------------------------------------------------------------


class StubOpenAICompat(BaseHTTPRequestHandler):
    """Canned /v1/models + /v1/chat/completions (OpenAI-compatible shape).
    mode: 'good' | 'short_batch'."""

    mode = "good"
    model_id = "gemma4-26b-a4b_q4"

    def log_message(self, *args):
        pass

    def _send(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        assert self.path == "/v1/models"
        self._send(
            {"object": "list", "data": [{"id": self.model_id, "object": "model"}]}
        )

    def do_POST(self):  # noqa: N802
        assert self.path == "/v1/chat/completions"
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        assert payload["stream"] is False
        assert "temperature" in payload
        if "model" in payload:
            assert payload["model"] == StubOpenAICompat.model_id
        user = payload["messages"][-1]["content"]
        numbered = re.findall(r"^(\d+)\s*:", user, flags=re.MULTILINE)
        if numbered:
            if StubOpenAICompat.mode == "short_batch":
                # Deliberately return fewer lines than asked -> forces fallback.
                content = "1: truncated-batch"
            else:
                content = "\n".join(f"{n}: english-{n}" for n in numbered)
        else:
            content = "OK-single"
        self._send(
            {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "done": True,
            }
        )


def _serve_endpoint() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubOpenAICompat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_srt_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.ja.srt")
        write_srt(FAKE_SEGMENTS, path)
        back = read_srt(path)
        assert len(back) == len(FAKE_SEGMENTS), f"got {len(back)}"
        for orig, got in zip(FAKE_SEGMENTS, back):
            assert abs(orig["start"] - got["start"]) < 0.0011, (orig, got)
            assert abs(orig["end"] - got["end"]) < 0.0011, (orig, got)
            assert orig["text"] == got["text"], (orig, got)
        # translated writer keeps timestamps 1:1
        tr_path = os.path.join(tmp, "out.english.srt")
        write_translated_srt(FAKE_SEGMENTS, ["a", "b", "c"], tr_path)
        tr_back = read_srt(tr_path)
        assert [e["text"] for e in tr_back] == ["a", "b", "c"]
        assert all(
            abs(a["start"] - b["start"]) < 0.0011 and abs(a["end"] - b["end"]) < 0.0011
            for a, b in zip(FAKE_SEGMENTS, tr_back)
        )
    print("PASS test_srt_roundtrip")


def test_endpoint_batch_1to1() -> None:
    server, endpoint = _serve_endpoint()
    try:
        StubOpenAICompat.mode = "good"
        StubOpenAICompat.model_id = "gemma4-26b-a4b_q4"
        backend = OpenAICompatBackend(
            endpoint=endpoint, model="gemma4-26b-a4b_q4", context_window=5, verify=True
        )
        texts = [f"text number {i}" for i in range(25)]  # spans 3 batches
        results = backend.translate_batch(texts)
        assert len(results) == len(texts), "not 1:1 length"
        assert all(r for r in results), f"unexpected empty results: {results}"
        # stub echoes batch-local numbering: 1..10, 1..10, 1..5
        assert results[0] == "english-1" and results[9] == "english-10"
        assert results[10] == "english-1" and results[24] == "english-5"
    finally:
        server.shutdown()
    print("PASS test_endpoint_batch_1to1")


def test_endpoint_empty_model_skips_verify_and_omits_field() -> None:
    server, endpoint = _serve_endpoint()
    try:
        StubOpenAICompat.mode = "good"
        backend = OpenAICompatBackend(endpoint=endpoint, model="", verify=True)
        results = backend.translate_batch(["aa", "bb"])
        assert results == ["english-1", "english-2"], results
        # (stub asserts `model` absent from payload when configured empty)
    finally:
        server.shutdown()
    print("PASS test_endpoint_empty_model_skips_verify_and_omits_field")


def test_endpoint_per_segment_fallback() -> None:
    server, endpoint = _serve_endpoint()
    try:
        StubOpenAICompat.mode = "short_batch"
        backend = OpenAICompatBackend(endpoint=endpoint, model="gemma4-26b-a4b_q4")
        results = backend.translate_batch(["one", "two", "three"])
        assert len(results) == 3, "not 1:1 length"
        assert all(r == "OK-single" for r in results), results
    finally:
        server.shutdown()
    print("PASS test_endpoint_per_segment_fallback")


def test_endpoint_unreachable_endpoint_error() -> None:
    try:
        OpenAICompatBackend(endpoint="http://127.0.0.1:1", model="anything.gguf")
    except BackendError as exc:
        assert "Cannot reach the OpenAI-compatible endpoint" in str(exc), str(exc)
    else:
        raise AssertionError("expected BackendError")
    print("PASS test_endpoint_unreachable_endpoint_error")


def test_endpoint_model_not_found_error() -> None:
    server, endpoint = _serve_endpoint()
    try:
        StubOpenAICompat.model_id = "gemma4-26b-a4b_q4"
        try:
            OpenAICompatBackend(endpoint=endpoint, model="wrong-model.gguf")
        except ModelNotFoundError as exc:
            msg = str(exc)
            assert "wrong-model.gguf" in msg, msg
            assert "gemma4-26b-a4b_q4" in msg, msg  # lists available ids
            assert 'model=""' in msg, msg
        else:
            raise AssertionError("expected ModelNotFoundError")
    finally:
        server.shutdown()
    print("PASS test_endpoint_model_not_found_error")


def main() -> int:
    test_srt_roundtrip()
    test_endpoint_batch_1to1()
    test_endpoint_empty_model_skips_verify_and_omits_field()
    test_endpoint_per_segment_fallback()
    test_endpoint_unreachable_endpoint_error()
    test_endpoint_model_not_found_error()
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
