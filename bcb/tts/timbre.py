"""Gnani Timbre text-to-speech client.

Verified against https://docs.gnani.ai/api/TTS/tts-inference.md and
https://docs.gnani.ai/api/TTS/available-voices.md on 5 November 2026:

  * ``POST https://api.vachana.ai/api/v1/tts/inference``, ``application/json``
  * auth header ``X-API-Key-ID: <key>``
  * body ``{text, model, voice, language, speed, audio_config}``
  * ``model`` is required: ``timbre-v3.0`` or ``timbre-v2.5``
  * the response body is **binary audio**, not base64 JSON
  * ``audio_config`` is required; asking for
    ``{sample_rate: 16000, encoding: linear_pcm, container: wav}`` gives a wav
    the benchmark can read with soundfile and degrade itself — no ffmpeg, and
    no second lossy conversion before the telephony chain.

Timbre v3.0 shares 13 voices across all ten languages; v2.5 has 42 voices each
tied to a preferred language. Voices are **not** interchangeable between the
two models, so the voice map is per model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import httpx

PATH = "/api/v1/tts/inference"


def base() -> str:
    """Base URL, read per request so GNANI_BASE_URL can be set at runtime."""
    return os.getenv("GNANI_BASE_URL", "https://api.vachana.ai").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5

#: Timbre v3.0: every voice speaks every supported language.
V3_VOICES = {
    "female": ["Shivani", "Lakshmi", "Kalpana", "Chetana", "Vasudha", "Riyaa", "Tarika"],
    "male": ["Avinash", "Shubham", "Prakash", "Devansh", "Yashraj", "Harshad"],
}

#: Timbre v2.5: voices are listed under the language they are tuned for.
V25_VOICES_BY_LANG = {
    "en-IN": ["Kaveri", "Trupti", "Devika", "Pranav", "Shlok", "Girish"],
    "hi-IN": ["Nalini", "Bhavna", "Yashvi", "Urmila", "Jwala", "Chitra", "Ambuja",
              "Deepak", "Roopesh", "Vikrant", "Hemraj", "Jalaj", "Omkar"],
    "hi-en": ["Poorvi"],
    "ta-IN": ["Asmita", "Trisha", "Brinda", "Vedika", "Noopur"],
    "te-IN": ["Suhana", "Lehara", "Lavanya", "Yukti", "Varuni"],
    "kn-IN": ["Saanvi", "Kavin"],
    "ml-IN": ["Reshma", "Riyaan"],
    "mr-IN": ["Zahira", "Ishaan"],
    "bn-IN": ["Kirra", "Dhruva"],
    "gu-IN": ["Falak", "Veera"],
    "pa-IN": ["Mehuli", "Zayan"],
}

TTS_LANGS = {"auto", "hi-IN", "en-IN", "hi-en", "ta-IN", "te-IN", "kn-IN",
             "ml-IN", "mr-IN", "pa-IN", "bn-IN", "gu-IN"}

#: `mulaw` and `alaw` containers are forced to 8 kHz by the API.
FORCED_8K_CONTAINERS = {"mulaw", "alaw"}


class TimbreError(RuntimeError):
    pass


def voices_for(lang: str, model: str = "timbre-v3.0") -> list[str]:
    """Voices usable for `lang`, in a stable order.

    For v3.0 this is every voice (female first, then male) since all voices
    speak all languages. For v2.5 it is the voices tuned for that language.
    """
    if model == "timbre-v3.0":
        return V3_VOICES["female"] + V3_VOICES["male"]
    return list(V25_VOICES_BY_LANG.get(lang, []))


def pick_voice(lang: str, key: str, model: str = "timbre-v3.0") -> str:
    """Deterministically choose a voice for `key` (e.g. a call_id).

    Rotating voices across calls keeps the benchmark from measuring one
    speaker's idiosyncrasies, and the same call always gets the same voice so
    reruns are reproducible.
    """
    pool = voices_for(lang, model)
    if not pool:
        raise TimbreError(f"no {model} voice for language {lang!r}")
    index = int(hashlib.sha1(f"{model}|{lang}|{key}".encode()).hexdigest()[:8], 16)
    return pool[index % len(pool)]


def audio_config(
    sample_rate: int = 16000,
    encoding: str = "linear_pcm",
    container: str = "wav",
    num_channels: int = 1,
    sample_width: int = 2,
) -> dict[str, Any]:
    if container in FORCED_8K_CONTAINERS:
        sample_rate = 8000
    return {
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "sample_width": sample_width,
        "encoding": encoding,
        "container": container,
    }


def _describe(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    for getter in (
        lambda b: b["message"],
        lambda b: f"{b['detail']['error_code']}: {b['detail']['message']}",
        lambda b: b["detail"],
    ):
        try:
            value = getter(body)
        except (KeyError, TypeError):
            continue
        if value:
            return f"HTTP {response.status_code}: {value}"
    return f"HTTP {response.status_code}: {json.dumps(body)[:200]}"


def synthesize(
    text: str,
    lang: str,
    voice: str,
    model: str = "timbre-v3.0",
    speed: float | str = 1.0,
    config: Optional[dict] = None,
) -> bytes:
    """Return the raw audio bytes Timbre produced for `text`."""
    if lang not in TTS_LANGS:
        raise TimbreError(
            f"Timbre does not support language {lang!r}; expected one of {sorted(TTS_LANGS)}"
        )
    key = os.getenv("GNANI_API_KEY")
    if not key:
        raise TimbreError("GNANI_API_KEY is not set (see .env.example)")
    body = {
        "text": text,
        "model": model,
        "voice": voice,
        "language": lang,
        "speed": speed,
        "audio_config": config or audio_config(),
    }
    last: Optional[Exception] = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = httpx.post(
                base() + PATH,
                headers={"X-API-Key-ID": key, "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            if not r.content:
                raise TimbreError("Timbre returned an empty body")
            return r.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS - 1:
                raise TimbreError(_describe(exc.response)) from exc
            last = exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            last = exc
        time.sleep(min(2**attempt + random.random(), 30))
    raise TimbreError(str(last))


def synthesize_to_file(
    text: str,
    lang: str,
    dst: Path,
    voice: Optional[str] = None,
    model: str = "timbre-v3.0",
    speed: float | str = 1.0,
    sample_rate: int = 16000,
) -> str:
    """Synthesise straight to a 16-bit mono wav at `dst`. Returns the voice used."""
    voice = voice or pick_voice(lang, dst.stem, model)
    data = synthesize(
        text, lang, voice, model=model, speed=speed,
        config=audio_config(sample_rate=sample_rate),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return voice
