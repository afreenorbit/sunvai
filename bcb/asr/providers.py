"""Speech-to-text clients behind one interface, with an on-disk response cache.

Endpoints were confirmed against each provider's live documentation on
**5 November 2026**; each class records what was checked. Re-verify before a
publication run and update the date.

Every request is cached by sha256(provider config + audio bytes), so re-running
never re-bills. Delete .cache/asr/<provider>/ to force a fresh call.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Optional

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=10.0)

#: Status codes worth retrying. 429 is Gnani's per-account concurrency limit,
#: which on the hackathon tier can be a single in-flight request.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


class ProviderError(RuntimeError):
    """An API-level failure, recorded against the clip and excluded from scoring."""


class ASRProvider(ABC):
    name: str

    #: Longest clip the API accepts, in seconds. None = no documented limit.
    max_audio_seconds: Optional[float] = None

    def __init__(self, name: str, cache_dir: str = ".cache/asr", **params: Any) -> None:
        self.name = name
        self.params = params
        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API -------------------------------------------------------- #
    def transcribe(self, wav_path: str, lang: str) -> dict:
        """Return {'text', 'latency_ms', 'raw', 'error', 'cached'}.

        Never raises on API errors: the error string is recorded so the scorer
        can exclude the clip instead of treating a failure as an empty
        transcript.
        """
        audio = Path(wav_path).read_bytes()
        key = self.cache_key(audio, lang)
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            out = json.loads(cached.read_text(encoding="utf-8"))
            out["cached"] = True
            return out

        t0 = time.perf_counter()
        try:
            raw = self._request_with_retries(audio, Path(wav_path).name, lang)
            text, error = self._extract_text(raw), None
        except Exception as exc:  # recorded, never silently estimated
            raw, text, error = None, "", f"{type(exc).__name__}: {exc}"
        out = {
            "text": text,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "raw": raw,
            "error": error,
        }
        if error is None:
            cached.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        out["cached"] = False
        return out

    def cache_key(self, audio: bytes, lang: str) -> str:
        """Cache identity: provider name + every request parameter + the audio."""
        meta = json.dumps(
            {"p": self.name, "client": type(self).__name__, "params": self.params, "lang": lang},
            sort_keys=True,
        )
        return hashlib.sha256(meta.encode() + audio).hexdigest()

    # ---- shared HTTP ------------------------------------------------------- #
    def _request_with_retries(self, audio: bytes, filename: str, lang: str) -> Any:
        last: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._request(audio, filename, lang)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS - 1:
                    raise ProviderError(self._describe(exc.response)) from exc
                last = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                last = exc
            # No Retry-After is sent, so back off exponentially with jitter.
            time.sleep(min(2**attempt + random.random(), 30))
        raise ProviderError(str(last))

    @staticmethod
    def _describe(response: httpx.Response) -> str:
        """Readable one-liner from any of the provider error envelopes."""
        try:
            body = response.json()
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"
        for getter in (
            # Keep the machine-readable type: a run log that says
            # MAX_AUDIO_DURATION_EXCEEDED is actionable, "bad request" is not.
            lambda b: f"{b['error']['type']}: {b['error']['message']}",
            lambda b: b["error"]["message"],
            lambda b: f"{b['detail']['error_code']}: {b['detail']['message']}",
            lambda b: b["message"],
            lambda b: b["detail"],
            lambda b: b["error"],
        ):
            try:
                value = getter(body)
            except (KeyError, TypeError):
                continue
            if value:
                return f"HTTP {response.status_code}: {value}"
        return f"HTTP {response.status_code}: {json.dumps(body)[:200]}"

    # ---- provider-specific ------------------------------------------------- #
    @abstractmethod
    def _request(self, audio: bytes, filename: str, lang: str) -> Any: ...

    @abstractmethod
    def _extract_text(self, raw: Any) -> str: ...


def _key(env: str) -> str:
    value = os.getenv(env)
    if not value:
        raise ProviderError(f"{env} is not set (see .env.example)")
    return value


# --------------------------------------------------------------------------- #
# Gnani Prisma
# --------------------------------------------------------------------------- #

#: Default base URL for every Gnani speech API.
GNANI_DEFAULT_BASE = "https://api.vachana.ai"


def gnani_base() -> str:
    """Base URL, read per request so GNANI_BASE_URL can be set at runtime.

    Reading it at import time would mean a process that sets the variable
    later — a test pointing at the mock server, a notebook switching to a
    staging host — silently kept calling production.
    """
    return os.getenv("GNANI_BASE_URL", GNANI_DEFAULT_BASE).rstrip("/")

#: Output formats. `verbatim` is raw spoken form; `transcribe` applies inverse
#: text normalisation (so "tees lakh" comes back as digits); `codemix` keeps
#: English in Latin script and `transliterate` romanises the whole transcript.
#: The last two are Prisma v3.0 only.
GNANI_FORMATS = {"verbatim", "transcribe", "codemix", "transliterate"}
GNANI_V3_ONLY_FORMATS = {"codemix", "transliterate"}

GNANI_LANGS = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN",
    "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN",
}


class GnaniPrisma(ASRProvider):
    """Gnani Prisma speech-to-text, REST.

    Verified against https://docs.gnani.ai/api/STT/speech-to-text.md on
    5 November 2026 (that page's own "last verified" line reads 5 August 2026):

      * ``POST https://api.vachana.ai/stt/v3``, ``multipart/form-data``
      * auth header ``X-API-Key-ID: <key>``
      * form fields ``audio_file``, ``language_code``, ``model``, ``format``
      * transcript at the top level, in ``transcript``
      * Prisma v2.5 is the default and is selected by **omitting** ``model``;
        sending ``model=prisma-v2.5`` explicitly is also accepted.
      * 30 s maximum audio duration

    Config params (all optional except where noted):
      model            prisma-v3.0 | prisma-v2.5
      format           verbatim | transcribe | codemix | transliterate
      language_code    overrides the per-clip language (e.g. "indic-auto" or
                       "hi-IN,en-IN" on v3.0)
      itn_native_numerals, keyterms, topic, previous_text, speaker_names,
      enable_substitution, substitution_map   (v3.0)
      bias_list, bias_score                   (v2.5 word boosting)
    """

    PATH = "/stt/v3"
    max_audio_seconds = 30.0

    #: Fields passed straight through to the form when present in the config.
    PASSTHROUGH = (
        "itn_native_numerals",
        "keyterms",
        "topic",
        "previous_text",
        "speaker_names",
        "enable_substitution",
        "substitution_map",
        "bias_list",
        "bias_score",
    )

    def __init__(self, name: str, cache_dir: str = ".cache/asr", **params: Any) -> None:
        super().__init__(name, cache_dir=cache_dir, **params)
        model = params.get("model")
        fmt = params.get("format")
        if model and model not in ("prisma-v3.0", "prisma-v2.5"):
            raise ValueError(f"unknown Prisma model {model!r}")
        if fmt and fmt not in GNANI_FORMATS:
            raise ValueError(f"unknown Prisma format {fmt!r}, expected one of {sorted(GNANI_FORMATS)}")
        # v2.5 answers INVALID_FORMAT for these, which would waste a whole run.
        if fmt in GNANI_V3_ONLY_FORMATS and model != "prisma-v3.0":
            raise ValueError(f"format={fmt!r} requires model=prisma-v3.0 (got {model!r})")

    @property
    def url(self) -> str:
        return gnani_base() + self.PATH

    def _form(self, lang: str) -> dict[str, str]:
        p = self.params
        form: dict[str, str] = {}
        # `transliterate` on v3.0 is the one case where language_code is optional.
        language = p.get("language_code", lang)
        if language:
            form["language_code"] = language
        if p.get("model"):
            form["model"] = p["model"]
        if p.get("format"):
            form["format"] = p["format"]
        for field in self.PASSTHROUGH:
            if p.get(field) is not None:
                value = p[field]
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                form[field] = str(value)
        return form

    def _request(self, audio, filename, lang):
        r = httpx.post(
            self.url,
            headers={"X-API-Key-ID": _key("GNANI_API_KEY")},
            data=self._form(lang),
            files={"audio_file": (filename, audio, "audio/wav")},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def _extract_text(self, raw):
        if isinstance(raw, dict) and raw.get("success") is False:
            err = raw.get("error") or {}
            raise ProviderError(f"{err.get('type', 'ERROR')}: {err.get('message', raw)}")
        # v3.0 and v2.5 both return a top-level `transcript`; v2.5 additionally
        # returns output.literal, the pre-ITN text.
        text = raw.get("transcript")
        if text is None:
            text = (raw.get("output") or {}).get("literal")
        if text is None:
            raise ProviderError(f"no transcript field in response: {sorted(raw)[:6]}")
        return text


# --------------------------------------------------------------------------- #
# Comparison providers
# --------------------------------------------------------------------------- #


class Sarvam(ASRProvider):
    """Sarvam speech-to-text. Verified against https://docs.sarvam.ai 5 Nov 2026."""

    URL = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")

    def _request(self, audio, filename, lang):
        data = {"language_code": self.params.get("language_code", lang)}
        if self.params.get("model"):
            data["model"] = self.params["model"]
        r = httpx.post(
            self.URL,
            headers={"api-subscription-key": _key("SARVAM_API_KEY")},
            data=data,
            files={"file": (filename, audio, "audio/wav")},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def _extract_text(self, raw):
        for field in ("transcript", "text"):
            if isinstance(raw.get(field), str):
                return raw[field]
        raise ProviderError(f"no transcript field in response: {sorted(raw)[:6]}")


class GroqWhisper(ASRProvider):
    """Whisper large-v3 via Groq's OpenAI-compatible endpoint.

    Included as a well-known open-weights baseline. Whisper takes a bare
    two-letter language code, not a BCP-47 tag.
    """

    URL = os.getenv("GROQ_STT_URL", "https://api.groq.com/openai/v1/audio/transcriptions")

    def _request(self, audio, filename, lang):
        r = httpx.post(
            self.URL,
            headers={"Authorization": f"Bearer {_key('GROQ_API_KEY')}"},
            data={
                "model": self.params.get("model", "whisper-large-v3"),
                "language": lang.split("-")[0],
                "response_format": "json",
            },
            files={"file": (filename, audio, "audio/wav")},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def _extract_text(self, raw):
        return raw["text"]


class LocalWhisper(ASRProvider):
    """Whisper running locally through faster-whisper. **No API key, no network.**

    This exists so the whole pipeline — and the extractor study that depends on
    realistic noisy transcripts — can run with no credentials at all. Install
    with ``pip install -e ".[local]"``; the first call downloads the weights.

    Two honest caveats, both of which the scorer and the docs must respect:

    * ``latency_ms`` here is local compute time on whatever machine ran it. It
      is **not** comparable to a hosted provider's latency and must never be
      put in the same latency column as one.
    * A small multilingual Whisper is weak on Kannada and Tamil. That is a
      feature for *generating* noisy training transcripts — the extractor study
      wants realistic recognition errors — and a liability for *ranking*
      providers. Use it for the former, and label it clearly in the latter.

    Config params: ``model`` (a faster-whisper size or HF id, default "small"),
    ``device``, ``compute_type``, ``beam_size``, ``vad_filter``.
    """

    #: Loaded models are shared across instances: the weights are large and
    #: several provider entries may use the same size.
    _LOADED: ClassVar[dict[tuple, Any]] = {}

    def _model(self):
        key = (
            self.params.get("model", "small"),
            self.params.get("device", "cpu"),
            self.params.get("compute_type", "int8"),
        )
        if key not in self._LOADED:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ProviderError(
                    'faster-whisper is not installed; pip install -e ".[local]"'
                ) from exc
            self._LOADED[key] = WhisperModel(key[0], device=key[1], compute_type=key[2])
        return self._LOADED[key]

    def _request(self, audio, filename, lang):
        import tempfile

        model = self._model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            path = tmp.name
        try:
            segments, info = model.transcribe(
                path,
                language=lang.split("-")[0] if lang and lang != "auto" else None,
                beam_size=self.params.get("beam_size", 5),
                vad_filter=self.params.get("vad_filter", False),
            )
            parts = [s.text for s in segments]
        finally:
            Path(path).unlink(missing_ok=True)
        return {
            "text": "".join(parts).strip(),
            "detected_language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "local": True,
        }

    def _extract_text(self, raw):
        return raw["text"]


CLIENTS: dict[str, type[ASRProvider]] = {
    "gnani_prisma": GnaniPrisma,
    "sarvam": Sarvam,
    "groq_whisper": GroqWhisper,
    "local_whisper": LocalWhisper,
}


def build(name: str, spec: dict, cache_dir: str = ".cache/asr") -> ASRProvider:
    spec = dict(spec)
    client = spec.pop("client")
    if client not in CLIENTS:
        raise ValueError(f"unknown ASR client {client!r}, expected one of {sorted(CLIENTS)}")
    # Keys that describe the entry rather than the REST request. Dropping them
    # here also keeps them out of the cache key, so adding a streaming client
    # to an existing entry does not invalidate its cached transcripts.
    for meta in ("label", "stream", "stream_sample_rate"):
        spec.pop(meta, None)
    return CLIENTS[client](name, cache_dir=cache_dir, **spec)


def build_all(cfg: dict, only: Optional[str] = None) -> dict[str, ASRProvider]:
    """Build every configured provider, or just the comma-separated `only` list."""
    if only in (None, "all"):
        names = list(cfg["asr"])
    else:
        names = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in names if n not in cfg["asr"]]
        if unknown:
            raise ValueError(f"not in config: {unknown}. Available: {sorted(cfg['asr'])}")
    return {n: build(n, cfg["asr"][n], cfg["paths"]["cache"]) for n in names}
