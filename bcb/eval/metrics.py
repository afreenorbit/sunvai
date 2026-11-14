"""WER, CER and Critical Error Rate (see docs/METRICS.md)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import jiwer
from rapidfuzz import fuzz

from bcb.eval import normalize as N
from bcb.schema import CriticalSpan

DEFAULT_WEIGHTS = {"amount": 3, "digits": 3, "consent": 2, "date": 2, "name": 1}


# --------------------------------------------------------------------------- #
# WER / CER (dataset level)
# --------------------------------------------------------------------------- #

def _prep(texts: Iterable[str], romanize: bool) -> list[str]:
    out = []
    for t in texts:
        t = N.romanize(t) if romanize else t
        out.append(N.for_wer(t) or "<empty>")
    return out


def wer(references: list[str], hypotheses: list[str], romanize: bool = True) -> float:
    """Dataset-level WER: total edits / total reference words."""
    return jiwer.wer(_prep(references, romanize), _prep(hypotheses, romanize))


def cer(references: list[str], hypotheses: list[str], romanize: bool = False) -> float:
    """Dataset-level CER. Use romanize=False for native-script references."""
    return jiwer.cer(_prep(references, romanize), _prep(hypotheses, romanize))


# --------------------------------------------------------------------------- #
# Critical Error Rate
# --------------------------------------------------------------------------- #

def span_correct(span: CriticalSpan, hypothesis: str, name_threshold: float = 0.85) -> bool:
    """Is this labelled critical span correctly present in the hypothesis?"""
    if span.type == "amount":
        return int(span.value) in N.parse_amounts(hypothesis)
    if span.type == "digits":
        target = str(span.value)
        return any(target in s for s in N.parse_digit_strings(hypothesis, min_len=1))
    if span.type == "consent":
        return N.consent_polarity(hypothesis) == span.value
    if span.type == "date":
        return str(span.value) in N.parse_dates(hypothesis)
    if span.type == "name":
        return name_similarity(str(span.value), hypothesis) >= name_threshold
    raise ValueError(f"unknown span type {span.type}")


def name_similarity(name: str, hypothesis: str) -> float:
    """How well every part of `name` appears in `hypothesis`, 0.0 to 1.0.

    Two deliberate choices, each fixing a flaw in the obvious implementation:

    1. **Both readings.** Romanising is necessary when a model answers in
       native script, but ITRANS restores the inherent schwa, so "प्रिया नायर"
       becomes "priya naayara" and "ரவி குமார்" becomes "ravi ghumar". Scoring
       the romanised form *only* marks a correctly transcribed name wrong, so
       the raw and romanised readings are both tried and the better one wins.
       A Latin hypothesis is unaffected.
    2. **Every token, not the best window.** The previous implementation used
       `partial_ratio` over the whole string, which is maximised by matching
       any substring — so "Priya Menon" scored as well against "Priya Nair" as
       a correct native-script transcription did. Matching the first name was
       enough. Each token of the target must now be found, and the per-token
       best matches are averaged.

    **Known limitation, measured not assumed.** On 17 hand-built cases the
    lowest correct score was 0.800 and the highest *incorrect* score 0.786 — a
    margin of 0.014. Correct-but-romanised names and wrong-but-similar names
    (Imran Khan vs Rohan Khan) overlap heavily, so no threshold separates them
    robustly. This is why `name` carries the lowest CrER weight (1, against 3
    for amount and digits), and why name misses are hand-audited rather than
    trusted. See docs/METRICS.md.
    """
    targets = N.basic(name).split()
    if not targets:
        return 0.0
    best = 0.0
    for reading in (N.basic(hypothesis), N.basic(N.romanize(hypothesis))):
        tokens = reading.split()
        if not tokens:
            continue
        per_token = [max(fuzz.ratio(t, h) / 100.0 for h in tokens) for t in targets]
        best = max(best, sum(per_token) / len(per_token))
    return best


@dataclass
class CrERResult:
    weighted: float
    per_type_miss_rate: dict[str, float]
    per_type_counts: dict[str, int]
    misses: list[dict] = field(default_factory=list)


def critical_error_rate(
    items: Iterable[tuple[str, list[CriticalSpan], str]],
    weights: Optional[dict[str, float]] = None,
    name_threshold: float = 0.85,
) -> CrERResult:
    """items: (clip_id, reference spans, hypothesis text)."""
    weights = weights or DEFAULT_WEIGHTS
    total_w = missed_w = 0.0
    counts: dict[str, int] = defaultdict(int)
    missed: dict[str, int] = defaultdict(int)
    misses: list[dict] = []

    for clip_id, spans, hyp in items:
        for span in spans:
            w = weights.get(span.type, 1)
            counts[span.type] += 1
            total_w += w
            if not span_correct(span, hyp, name_threshold):
                missed[span.type] += 1
                missed_w += w
                misses.append(
                    {"clip_id": clip_id, "type": span.type, "expected": span.value,
                     "surface": span.surface, "hypothesis": hyp}
                )

    per_type = {t: missed[t] / counts[t] for t in counts}
    return CrERResult(
        weighted=(missed_w / total_w) if total_w else 0.0,
        per_type_miss_rate=per_type,
        per_type_counts=dict(counts),
        misses=misses,
    )
