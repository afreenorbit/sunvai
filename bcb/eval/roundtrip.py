"""TTS -> STT round trip for Indian number formats.

    python -m bcb.eval.roundtrip --lang hi-IN

Writes a known value as text, has the TTS engine speak it, transcribes it, and
checks the parsed value survives. Results: results/roundtrip.csv
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

from bcb.asr.providers import build
from bcb.eval import normalize as N
from bcb.tts.synthesize import gnani_timbre

CASES = [
    # (format, text to speak, check, expected)
    ("crore_decimal", "Aapka loan ₹2.25 crore ka sanction hua hai.", "amount", 22_500_000),
    ("crore_grouped", "Total amount ₹2,25,00,000 hai.", "amount", 22_500_000),
    ("lakh", "EMI ke liye ₹3,50,000 chahiye.", "amount", 350_000),
    ("thousand", "Refund ₹25,000 ka process ho gaya.", "amount", 25_000),
    ("small", "Bas ₹499 ka recharge karna hai.", "amount", 499),
    ("digits_6", "Aapka order number 004417 hai.", "digits", "004417"),
    ("digits_4", "Last four digits 4417 hain.", "digits", "4417"),
    ("date_numeric", "Payment 14/11/2026 tak karna hai.", "date", "2026-11-14"),
    ("date_words", "Delivery 3 December 2026 ko hogi.", "date", "2026-12-03"),
]


def check(kind: str, text: str, expected) -> bool:
    if kind == "amount":
        return expected in N.parse_amounts(text)
    if kind == "digits":
        return any(expected in s for s in N.parse_digit_strings(text, min_len=1))
    return expected in N.parse_dates(text)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi-IN")
    ap.add_argument("--asr", default="prisma_v3_transcribe")
    ap.add_argument("--out", default="results/roundtrip.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/bench.yaml"))
    asr = build(args.asr, cfg["asr"][args.asr], cfg["paths"]["cache"])
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for fmt, text, kind, expected in CASES:
            wav = Path(d) / f"{fmt}.wav"
            gnani_timbre(text, args.lang, wav)
            hyp = asr.transcribe(str(wav), args.lang)["text"]
            rows.append([fmt, text, hyp, expected, check(kind, hyp, expected)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["format", "spoken_text", "transcript", "expected", "survived"])
        w.writerows(rows)
    print(f"{sum(r[-1] for r in rows)}/{len(rows)} values survived the round trip")


if __name__ == "__main__":
    main()
