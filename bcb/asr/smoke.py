"""Preflight: prove every configured API works before spending a run on it.

    python -m bcb.asr.smoke                  # Timbre -> every ASR provider
    python -m bcb.asr.smoke --wav clip.wav   # use your own audio instead
    python -m bcb.asr.smoke --lang kn-IN
    python -m bcb.asr.smoke --provider prisma_v3_transcribe,prisma_v25

With no --wav, this synthesises a line containing an amount and a digit string
with Gnani Timbre and sends that audio to every ASR provider. One command then
tells you three things: the TTS key works, the STT key works, and the critical
values survive the round trip.

Exit status is non-zero if anything failed, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import yaml
from dotenv import load_dotenv

from bcb.asr.providers import build_all
from bcb.eval import normalize as N

#: One line per language carrying an amount and a digit string, so the preflight
#: also shows whether critical values survive.
PROBES = {
    "hi-IN": ("Aapka loan teen lakh pachaas hazaar ka hai, account number char char ek saat.",
              350_000, "4417"),
    "kn-IN": ("Nimma loan mooru laksha ippattu saavira ide, account number naalku naalku ondu elu.",
              320_000, "4417"),
    "ta-IN": ("Ungal loan moonu latcham irupathu aayiram, account number naalu naalu onnu ezhu.",
              320_000, "4417"),
}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--wav", default=None, help="audio to send; default is a fresh Timbre clip")
    ap.add_argument("--lang", default="hi-IN", choices=sorted(PROBES))
    ap.add_argument("--provider", default="all", help="comma-separated provider names")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    text, want_amount, want_digits = PROBES[args.lang]
    failures = 0
    tmpdir = tempfile.mkdtemp(prefix="bcb-smoke-")

    # ---- 1. Timbre ------------------------------------------------------- #
    wav = args.wav
    if wav is None:
        from bcb.tts import timbre

        dst = Path(tmpdir) / f"probe_{args.lang}.wav"
        try:
            voice = timbre.synthesize_to_file(text, args.lang, dst)
            seconds = sf.info(str(dst)).duration
            print(f"[OK ] Timbre v3.0  voice={voice:10s} {seconds:4.1f}s  {dst}")
            wav = str(dst)
        except Exception as exc:
            print(f"[ERR] Timbre v3.0  {type(exc).__name__}: {exc}")
            print("\nCannot synthesise a probe clip. Check GNANI_API_KEY, or pass --wav.")
            return 1
    else:
        seconds = sf.info(wav).duration
        print(f"[OK ] input       {seconds:4.1f}s  {wav}")

    if seconds > 30:
        print(f"[WARN] clip is {seconds:.1f}s; Prisma rejects anything over 30s")

    # ---- 2. every ASR provider ------------------------------------------- #
    print(f"\nreference: {text}")
    print(f"expecting: amount={want_amount:,}  digits={want_digits}\n")

    cache = "/dev/null" if args.no_cache else cfg["paths"]["cache"]
    if args.no_cache:
        cache = tempfile.mkdtemp(prefix="bcb-smoke-cache-")
    cfg = {**cfg, "paths": {**cfg["paths"], "cache": cache}}

    try:
        providers = build_all(cfg, args.provider)
    except Exception as exc:
        print(f"[ERR] building providers: {exc}")
        return 1

    for name, provider in providers.items():
        try:
            res = provider.transcribe(wav, args.lang)
        except Exception as exc:
            print(f"[ERR] {name:24s} {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if res["error"]:
            print(f"[ERR] {name:24s} {res['error']}")
            failures += 1
            continue
        got_amount = want_amount in N.parse_amounts(res["text"])
        got_digits = any(want_digits in s for s in N.parse_digit_strings(res["text"], min_len=1))
        marks = f"{'A' if got_amount else '-'}{'D' if got_digits else '-'}"
        tag = "cache" if res["cached"] else f"{res['latency_ms']:.0f}ms"
        print(f"[OK ] {name:24s} {tag:>8s} [{marks}] {res['text']}")

    print(f"\nA = amount parsed correctly, D = digit string found, - = missed."
          f"\n{len(providers) - failures}/{len(providers)} providers responded.")
    if failures:
        print("\nFailures above are usually a missing key in .env or a provider not "
              "enabled on your account. Remove unused providers from configs/bench.yaml.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
