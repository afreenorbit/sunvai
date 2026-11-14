"""Build call scripts from templates — no LLM, no API key.

    python -m bcb.scripts.seed --config configs/bench.yaml --out data/scripts/seed.jsonl

`bcb.scripts.generate` writes richer, more varied turns with an LLM, and is the
path for the full dataset. This module exists so the pipeline has valid input
when no LLM credentials are available, and so the repository ships a small
reproducible set that anyone can rebuild offline with one command.

Every critical value is declared alongside the words that carry it, and
`tests/test_seed.py` asserts that the rule parsers in `bcb.eval.normalize`
recover each declared value from its own surface text. A template whose spoken
form the parsers cannot read is a bug in the template or a gap in a lexicon,
caught here rather than halfway down a benchmark run.

Call ids match `generate.py` (`{lang2}_{domain}_{index:03d}`) and splits use the
same stratified, seeded assignment, so seed and generated scripts can be mixed
in one run without a split leaking between train and test.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import yaml

from bcb.schema import CallScript, CriticalSpan, Turn
from bcb.scripts.generate import assign_splits

# --------------------------------------------------------------------------- #
# Critical values, with the exact words that carry them
# --------------------------------------------------------------------------- #
# (spoken surface, canonical value). The spoken form is what the TTS voice
# says, so it must be a natural way to speak the value in that language, and
# the parsers must be able to read the value back out of it.

AMOUNTS: dict[str, list[tuple[str, int]]] = {
    "hi-IN": [
        ("tees lakh", 3_000_000), ("teen lakh", 300_000),
        ("saadhe teen lakh", 350_000), ("dedh lakh", 150_000),
        ("dhai crore", 25_000_000), ("do lakh pachaas hazaar", 250_000),
        ("pachattar hazaar", 75_000), ("pandrah hazaar", 15_000),
        ("chaalis hazaar", 40_000), ("paanch sau", 500),
    ],
    "kn-IN": [
        ("muvattu laksha", 3_000_000), ("mooru laksha", 300_000),
        ("mooru laksha ippattu saavira", 320_000), ("eradu koti", 20_000_000),
        ("eppattu aidu saavira", 75_000), ("hattu saavira", 10_000),
        ("nalavattu saavira", 40_000), ("aidu nooru", 500),
    ],
    "ta-IN": [
        ("muppathu latcham", 3_000_000), ("moonu latcham", 300_000),
        ("moonu latcham irupathu aayiram", 320_000), ("rendu kodi", 20_000_000),
        ("ezhubathu aidu aayiram", 75_000), ("pathu aayiram", 10_000),
        ("naarpathu aayiram", 40_000), ("anju nooru", 500),
    ],
}

DIGIT_STRINGS: dict[str, list[tuple[str, str]]] = {
    "hi-IN": [
        ("char char ek saat", "4417"), ("do nau aath teen", "2983"),
        ("saat ek do nau", "7129"), ("nau nau char ek", "9941"),
        ("ek do teen char paanch chhe", "123456"),
    ],
    "kn-IN": [
        ("naalku naalku ondu elu", "4417"), ("eradu ombattu entu mooru", "2983"),
        ("elu ondu eradu ombattu", "7129"), ("ombattu ombattu naalku ondu", "9941"),
        ("ondu eradu mooru naalku aidu aaru", "123456"),
    ],
    "ta-IN": [
        ("naalu naalu onnu ezhu", "4417"), ("rendu onbadhu ettu moonu", "2983"),
        ("ezhu onnu rendu onbadhu", "7129"), ("onbadhu onbadhu naalu onnu", "9941"),
        ("onnu rendu moonu naalu anju aaru", "123456"),
    ],
}

DATES: list[tuple[str, str]] = [
    ("14 November 2026", "2026-11-14"), ("3 December 2026", "2026-12-03"),
    ("21 October 2026", "2026-10-21"), ("8 January 2027", "2027-01-08"),
    ("30 November 2026", "2026-11-30"),
]

CONSENT: dict[str, list[tuple[str, str]]] = {
    "hi-IN": [("haan ji", "yes"), ("bilkul theek hai", "yes"),
              ("nahi", "no"), ("nahi mujhe nahi chahiye", "no")],
    "kn-IN": [("houdu", "yes"), ("sari", "yes"),
              ("illa", "no"), ("beda", "no")],
    "ta-IN": [("aamaa", "yes"), ("seri", "yes"),
              ("illai", "no"), ("vendam", "no")],
}

# Romanised names, as the dataset card requires: no real customer identities.
# Tagged with grammatical gender because Hindi verbs agree with the speaker:
# "bol raha hoon" for a man, "bol rahi hoon" for a woman. Getting that wrong
# makes the line sound machine-written to any Hindi speaker — including the
# judges — and teaches the TTS voice the wrong prosody.
NAMES: list[tuple[str, str]] = [
    ("Rahul Sharma", "m"), ("Priya Nair", "f"), ("Arun Kumar", "m"),
    ("Sneha Reddy", "f"), ("Imran Khan", "m"), ("Kavya Iyer", "f"),
    ("Vikram Singh", "m"), ("Divya Menon", "f"), ("Suresh Babu", "m"),
    ("Anjali Desai", "f"),
]

#: Gendered verb forms for the {g} slot in a name frame, per language.
#: Kannada and Tamil name frames here are gender-neutral, so both map the same.
GENDER_FORMS: dict[str, dict[str, str]] = {
    "hi-IN": {"m": "raha", "f": "rahi"},
    "kn-IN": {"m": "", "f": ""},
    "ta-IN": {"m": "", "f": ""},
}

# --------------------------------------------------------------------------- #
# Sentence frames
# --------------------------------------------------------------------------- #
# {v} is replaced by the spoken surface of the critical value. The frame around
# it is ordinary code-mixed call language: English nouns, Indic grammar.

FRAMES: dict[str, dict[str, list[str]]] = {
    "hi-IN": {
        "name": ["Mera naam {v} hai.", "Ji, {v} bol {g} hoon."],
        "amount": ["{thing} ke liye {v} chahiye.", "Mujhe {v} ka {thing} chahiye tha.",
                   "{thing} ka amount {v} hai na?"],
        "digits": ["Mera {idnoun} {v} hai.", "{idnoun} note kar lijiye, {v}.",
                   "Last digits {v} hain."],
        "consent": ["{v}, aap aage badhiye.", "{v}.", "Dekhiye, {v}."],
        "date": ["{v} tak kar dunga.", "Date {v} rakh lijiye.",
                 "{v} ko payment ho jayega."],
    },
    "kn-IN": {
        "name": ["Nanna hesaru {v}.", "Ji, {v} maatadutha iddini."],
        "amount": ["{thing} ge {v} beku.", "Nange {v} {thing} beku ittu.",
                   "{thing} amount {v} alva?"],
        "digits": ["Nanna {idnoun} {v}.", "{idnoun} note maadkolli, {v}.",
                   "Last digits {v}."],
        "consent": ["{v}, munde hogi.", "{v}.", "Nodi, {v}."],
        "date": ["{v} olage maadtini.", "Date {v} itkolli.",
                 "{v} ge payment aagutte."],
    },
    "ta-IN": {
        "name": ["En peru {v}.", "Ji, {v} pesuren."],
        "amount": ["{thing} ku {v} venum.", "Enakku {v} {thing} venum.",
                   "{thing} amount {v} thaane?"],
        "digits": ["En {idnoun} {v}.", "{idnoun} note pannunga, {v}.",
                   "Last digits {v}."],
        "consent": ["{v}, mela poonga.", "{v}.", "Paarunga, {v}."],
        "date": ["{v} ulla pannidren.", "Date {v} vachukonga.",
                 "{v} ku payment aagidum."],
    },
}

#: The domain noun dropped into an {thing} slot, and the identifier in {idnoun}.
#: English loan words here are deliberate: that is how these calls are spoken.
DOMAIN_NOUNS = {
    "loan": "loan", "insurance": "policy", "delivery": "delivery",
    "kyc": "KYC", "recharge": "recharge",
}
ID_NOUNS = {
    "loan": "account number", "insurance": "policy number",
    "delivery": "order number", "kyc": "reference number",
    "recharge": "mobile number",
}

#: What the agent says before each customer turn. Kept short: agent turns are
#: context for the reader, and are not synthesised or scored.
AGENT_LINES = {
    "open": "Thank you for calling. May I have your name, and how can I help?",
    "amount": "Sure. What amount are we looking at?",
    "digits": "Could you confirm the number on the account?",
    "consent": "Do you consent to us processing this?",
    "date": "And by when can you complete it?",
}

#: One span type per customer turn, in this order, cycled to fill
#: `customer_turns_per_call`. The name rides along on the first customer turn
#: rather than taking a turn of its own — a caller really does say "I'm X and I
#: need Y", and at the default of 4 turns a separate name turn would crowd out
#: the `date` type entirely, leaving a whole CrER category unmeasured.
TURN_PLAN = ["amount", "digits", "consent", "date"]


def _sentence_case(text: str) -> str:
    """Capitalise the first letter only.

    A frame may begin with a slot, so a turn can otherwise start lower-case
    ("loan ke liye..."). `str.capitalize` would also lower-case the rest and
    destroy the English proper nouns and acronyms (KYC, EMI) these lines carry.
    """
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _value_for(kind: str, lang: str, rng: random.Random) -> tuple[str, object]:
    if kind == "amount":
        return rng.choice(AMOUNTS[lang])
    if kind == "digits":
        return rng.choice(DIGIT_STRINGS[lang])
    if kind == "consent":
        return rng.choice(CONSENT[lang])
    if kind == "date":
        return rng.choice(DATES)
    raise ValueError(f"no values for span type {kind!r}")


def build_call(call_id: str, lang: str, domain: str, n_turns: int, rng: random.Random) -> CallScript:
    """One script. Every critical value is declared as a span on its turn."""
    frames = FRAMES[lang]
    turns: list[Turn] = []
    index = 0
    plan = [TURN_PLAN[i % len(TURN_PLAN)] for i in range(n_turns)]

    for position, kind in enumerate(plan):
        surface, value = _value_for(kind, lang, rng)
        text = rng.choice(frames[kind]).format(
            v=surface, thing=DOMAIN_NOUNS[domain], idnoun=ID_NOUNS[domain]
        )
        spans = [CriticalSpan(type=kind, surface=surface, value=value)]

        if position == 0:
            # The caller introduces themselves and states their business in one
            # breath, so this turn carries both a name span and the plan's.
            name, gender = rng.choice(NAMES)
            intro = rng.choice(frames["name"]).format(
                v=name, g=GENDER_FORMS[lang][gender]
            )
            text = f"{_sentence_case(intro)} {_sentence_case(text)}"
            spans.insert(0, CriticalSpan(type="name", surface=name, value=name))
        else:
            text = _sentence_case(text)

        agent_line = AGENT_LINES["open"] if position == 0 else AGENT_LINES[kind]
        turns.append(Turn(index=index, speaker="agent", text=agent_line))
        index += 1
        turns.append(Turn(index=index, speaker="customer", text=text, spans=spans))
        index += 1

    return CallScript(call_id=call_id, lang=lang, domain=domain, turns=turns)


def build_all(cfg: dict, per_lang: Optional[int] = None, seed: int = 42) -> list[CallScript]:
    per_lang = per_lang or cfg["calls_per_language"]
    domains = cfg["domains"]
    n_turns = cfg["customer_turns_per_call"]
    scripts: list[CallScript] = []
    for lang in (entry["code"] for entry in cfg["languages"]):
        for i in range(per_lang):
            domain = domains[i % len(domains)]
            call_id = f"{lang[:2]}_{domain}_{i:03d}"
            # Seed per call, so adding calls never changes existing ones.
            rng = random.Random(f"{seed}|{call_id}")
            scripts.append(build_call(call_id, lang, domain, n_turns, rng))
    assign_splits(scripts, cfg["split"])
    return scripts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--out", default="data/scripts/seed.jsonl")
    ap.add_argument("--per-lang", type=int, default=None, help="override calls_per_language")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    scripts = build_all(cfg, args.per_lang, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in scripts:
            f.write(s.model_dump_json() + "\n")

    spans: dict[str, int] = {}
    customer_turns = 0
    for s in scripts:
        for t in s.customer_turns():
            customer_turns += 1
            for span in t.spans:
                spans[span.type] = spans.get(span.type, 0) + 1
    by_split: dict[str, int] = {}
    for s in scripts:
        by_split[s.split or "?"] = by_split.get(s.split or "?", 0) + 1

    print(f"wrote {len(scripts)} scripts ({customer_turns} customer turns) to {out}")
    print(f"  splits: {dict(sorted(by_split.items()))}")
    print(f"  spans:  {dict(sorted(spans.items()))}")


if __name__ == "__main__":
    main()
