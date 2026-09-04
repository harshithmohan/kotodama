"""Stage 2: JA->EN translation via an OpenAI-compatible chat completions API.

Backend interface:
    translate_batch(texts: list[str], source_lang="ja", target_lang="en") -> list[str]

Contract: output is 1:1 with input (same length). Individual failures yield an
empty string for that segment and are logged — never a dropped segment.

Talks to the endpoint's OpenAI-compatible routes: POST {endpoint}/v1/chat/completions
(non-streaming, reads choices[0].message.content) and GET {endpoint}/v1/models for
startup verification. The endpoint serves a single model loaded at server
startup, so an empty configured `model` is valid (the model field is omitted
and the verification check is skipped).
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional

import requests

log = logging.getLogger("kotodama.translate")

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional subtitle translator. Translate Japanese subtitles "
    "into natural English in subtitle register: concise, easy to read aloud, "
    "matching the tone and context of the conversation. Preserve speaker intent; "
    "do not add explanations or commentary. Never mix languages."
)

# Number of segments translated per batched chat request.
BATCH_SIZE = 10

_NUMBERED_LINE = re.compile(r"^\s*(\d{1,4})\s*[:\uFF1A.]\s*(.*)$")


class BackendError(RuntimeError):
    """Backend configuration/connection problem the user must fix."""


class ModelNotFoundError(BackendError):
    """Configured model is not available on the endpoint."""


class OpenAICompatBackend:
    """Translation via an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        context_window: int = 5,
        request_timeout: float = 300.0,
        temperature: float = 0.2,
        verify: bool = True,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = (model or "").strip()
        self.context_window = max(0, int(context_window))
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.session = requests.Session()
        if verify:
            self.verify_model()

    # -- startup checks ---------------------------------------------------------

    def verify_model(self) -> None:
        """GET {endpoint}/v1/models. Skipped entirely when no model configured."""
        if not self.model:
            return  # whatever the server loaded is used
        url = f"{self.endpoint}/v1/models"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            ids = [
                m.get("id", "")
                for m in resp.json().get("data", [])
                if isinstance(m, dict)
            ]
        except requests.ConnectionError as exc:
            raise BackendError(
                f"Cannot reach the OpenAI-compatible endpoint at {self.endpoint}. "
                f"Start the server or point --endpoint / KOTODAMA_ENDPOINT at "
                f"your endpoint host. (underlying error: {exc})"
            ) from exc
        except Exception as exc:
            raise BackendError(f"Failed to query {url}: {exc}") from exc

        if self.model not in ids:
            listing = "\n  ".join(ids) if ids else "(none reported)"
            raise ModelNotFoundError(
                f"Model '{self.model}' is not served by the endpoint at "
                f"{self.endpoint}.\nAvailable model ids:\n  {listing}\n"
                f"Note: the endpoint serves a single model loaded at startup; "
                f"either restart the server with the desired model, set "
                f'[translate].model to a listed id, or leave model="" in '
                f"config to use whatever the server has loaded."
            )

    # -- request plumbing ---------------------------------------------------------

    def _chat(self, user_content: str) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        payload: dict = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        # Only send `model` when configured; the endpoint serves its single
        # loaded model, so omitting an empty value is safest.
        if self.model:
            payload["model"] = self.model
        resp = self.session.post(url, json=payload, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or [{}]
        return ((choices[0].get("message") or {}).get("content", "")).strip()

    # -- prompt construction --------------------------------------------------

    def _context_block(self, texts: list[str], lo: int, hi: int) -> str:
        """Surrounding-source-text context lines for window [lo, hi)."""
        start = max(0, lo - self.context_window)
        end = min(len(texts), hi + self.context_window)
        parts = []
        if start < lo:
            parts.append(
                "Context from before (already-seen Japanese, for reference only):\n"
                + "\n".join(texts[start:lo])
            )
        if end > hi:
            parts.append(
                "Context from after (upcoming Japanese, for reference only):\n"
                + "\n".join(texts[hi:end])
            )
        return "\n\n".join(parts)

    def _build_batch_prompt(self, texts: list[str], lo: int, hi: int) -> str:
        batch = texts[lo:hi]
        numbered = "\n".join(f"{i + 1}: {t}" for i, t in enumerate(batch))
        context = self._context_block(texts, lo, hi)
        instruction = (
            f"Translate the following {len(batch)} numbered Japanese subtitle "
            "segments into English. Return exactly one line per segment in the "
            "form 'N: <translation>' with the same numbering, no blank lines, "
            "no other text.\n\n"
        )
        prompt = instruction + numbered
        if context:
            prompt += "\n\n" + context
        return prompt

    def _build_single_prompt(self, texts: list[str], idx: int) -> str:
        context = self._context_block(texts, idx, idx + 1)
        instruction = (
            "Translate the following Japanese subtitle segment into natural "
            "English for subtitles. Return ONLY the translation, nothing else."
            f"\n\nJapanese: {texts[idx]}"
        )
        return instruction + (("\n\n" + context) if context else "")

    @staticmethod
    def _parse_numbered(content: str, count: int) -> Optional[list[str]]:
        found: dict[int, str] = {}
        for line in content.splitlines():
            m = _NUMBERED_LINE.match(line)
            if m:
                n = int(m.group(1))
                if 1 <= n <= count:
                    found[n] = m.group(2).strip()
        if len(found) == count and all(i in found for i in range(1, count + 1)):
            return [found[i] for i in range(1, count + 1)]
        return None

    # -- public interface ------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "ja",
        target_lang: str = "en",
    ) -> list[str]:
        n = len(texts)
        results: list[str] = [""] * n
        for lo in range(0, n, BATCH_SIZE):
            hi = min(lo + BATCH_SIZE, n)
            try:
                content = self._chat(self._build_batch_prompt(texts, lo, hi))
                parsed = self._parse_numbered(content, hi - lo)
            except Exception as exc:
                log.warning("batch request failed (%s); falling back per segment", exc)
                parsed = None
            if parsed is None:
                print(
                    f"[stage2] batch {lo + 1}-{hi} response mismatch, "
                    "retrying per-segment",
                    file=sys.stderr,
                )
                for i in range(lo, hi):
                    try:
                        results[i] = self._chat(self._build_single_prompt(texts, i))
                    except Exception as exc:
                        log.error("segment %d translation failed: %s", i + 1, exc)
            else:
                for offset, text in enumerate(parsed):
                    results[lo + offset] = text
            print(
                f"[stage2] translated {min(hi, n)}/{n} segments",
                file=sys.stderr,
            )
        return results
