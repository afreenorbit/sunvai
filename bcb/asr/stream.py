"""Streaming speech-to-text clients, for first-word / final latency measurement.

Verified against https://docs.gnani.ai/api/STT/stt-websocket.md on
5 November 2026:

  * ``wss://api.vachana.ai/stt/v3/stream``
  * configuration travels as **WebSocket upgrade headers**, not as a first JSON
    message: ``x-api-key-id`` (lower-case here), ``language_code``, ``model``,
    ``format``, ``x-sample-rate``, ``x-idle-timeout-sec``
  * audio frames are binary, raw PCM **signed 16-bit little-endian, mono**, and
    each frame must be **exactly 1024 bytes** (512 samples) — 32 ms at 16 kHz,
    64 ms at 8 kHz — sent at real-time cadence
  * ``{"type": "commit"}`` closes the open utterance
  * server messages are JSON with a ``type``: ``connected``, ``processing``,
    ``transcript`` (text in ``text``, with ``is_final``), ``error``
  * 15-minute session cap; 60 s default idle timeout

The frame size is a protocol requirement, not a tuning knob: a client that
sends 20 ms frames is rejected, so each client declares its own
``frame_bytes`` and the measurement loop paces itself accordingly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol

#: Gnani's realtime endpoint requires exactly this many bytes per binary frame.
GNANI_FRAME_BYTES = 1024


def _ws_base() -> str:
    base = os.getenv("GNANI_BASE_URL", "https://api.vachana.ai").rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):]
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):]
    return base


class StreamingClient(Protocol):
    """What `bcb.eval.latency` needs from a streaming provider."""

    frame_bytes: int
    sample_rate: int

    async def connect(self, lang: str) -> Any: ...
    async def send_frame(self, ws: Any, pcm: bytes) -> None: ...
    async def end_stream(self, ws: Any) -> None: ...
    def parse_message(self, msg: Any) -> tuple[str, bool]: ...


async def _ws_connect(url: str, headers: dict[str, str]):
    """Open a WebSocket, tolerating the `websockets` header-kwarg rename.

    websockets >= 14 takes `additional_headers`; 10–13 took `extra_headers`.
    """
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - declared in pyproject
        raise RuntimeError("pip install 'websockets>=12' to measure streaming latency") from exc

    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


class GnaniPrismaStream:
    """Prisma realtime streaming over WebSocket."""

    PATH = "/stt/v3/stream"
    frame_bytes = GNANI_FRAME_BYTES

    def __init__(
        self,
        model: str = "prisma-v3.0",
        fmt: str = "transcribe",
        sample_rate: int = 16000,
        idle_timeout_sec: int = 60,
        language_code: Optional[str] = None,
        **_ignored: Any,
    ) -> None:
        if sample_rate not in (8000, 16000, 44100, 48000):
            raise ValueError(f"Prisma streaming does not accept sample_rate={sample_rate}")
        self.model = model
        self.fmt = fmt
        self.sample_rate = sample_rate
        self.idle_timeout_sec = idle_timeout_sec
        self.language_code = language_code

    @property
    def url(self) -> str:
        return _ws_base() + self.PATH

    async def connect(self, lang: str):
        key = os.getenv("GNANI_API_KEY")
        if not key:
            raise RuntimeError("GNANI_API_KEY is not set (see .env.example)")
        headers = {
            "x-api-key-id": key,
            "language_code": self.language_code or lang,
            "model": self.model,
            "format": self.fmt,
            "x-sample-rate": str(self.sample_rate),
            "x-idle-timeout-sec": str(self.idle_timeout_sec),
        }
        return await _ws_connect(self.url, headers)

    async def send_frame(self, ws, pcm: bytes) -> None:
        # The server rejects short frames, so pad the file's final partial frame.
        if len(pcm) < self.frame_bytes:
            pcm = pcm + b"\x00" * (self.frame_bytes - len(pcm))
        await ws.send(pcm)

    async def end_stream(self, ws) -> None:
        await ws.send(json.dumps({"type": "commit"}))

    def parse_message(self, msg) -> tuple[str, bool]:
        """Return (text, is_final). Non-transcript messages yield ("", False)."""
        if isinstance(msg, (bytes, bytearray)):
            return "", False
        try:
            data = json.loads(msg)
        except (TypeError, ValueError):
            return "", False
        kind = data.get("type")
        if kind == "error":
            raise RuntimeError(f"stream error: {data.get('message')}")
        if kind != "transcript":
            return "", False
        return data.get("text") or "", bool(data.get("is_final"))


#: Streaming clients, keyed by the `stream` name used in configs/bench.yaml.
CLIENTS: dict[str, type] = {"gnani_prisma": GnaniPrismaStream}


def build_stream(spec: dict) -> StreamingClient:
    """Build a streaming client from a provider's config entry.

    The entry needs a `stream:` key naming the client; the ASR `model` and
    `format` are reused so streaming is measured on the same configuration
    that is scored for accuracy.
    """
    spec = dict(spec)
    name = spec.pop("stream", None)
    if not name:
        raise ValueError("this provider has no `stream:` client configured")
    if name not in CLIENTS:
        raise ValueError(f"unknown streaming client {name!r}, expected one of {sorted(CLIENTS)}")
    kwargs = {
        "model": spec.get("model", "prisma-v3.0"),
        "fmt": spec.get("format", "transcribe"),
        "sample_rate": spec.get("stream_sample_rate", 16000),
        "language_code": spec.get("language_code"),
    }
    return CLIENTS[name](**kwargs)
