"""Text normalisation and critical-value parsing for code-mixed Indian speech.

Handles:
  * Unicode NFC, punctuation, whitespace
  * romanisation of Devanagari / Kannada / Tamil text (for WER-rom and name matching)
  * Indian amounts in digits ("₹2.25 crore", "2,25,00,000", "30L") and in words
    (English, romanised Hindi / Kannada / Tamil, common Devanagari number words)
  * spoken digit strings ("char char ek saat", "double four one seven")
  * consent polarity (yes / no / none)
  * simple dates ("14/11/2026", "14 November 2026")

The lexicons are deliberately explicit so reviewers can audit them. Extend them
when the error gallery shows a missed form — and add a test for it.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

# --------------------------------------------------------------------------- #
# Basic normalisation
# --------------------------------------------------------------------------- #

_NATIVE_DIGITS = str.maketrans(
    "०१२३४५६७८९" "೦೧೨೩೪೫೬೭೮೯" "௦௧௨௩௪௫௬௭௮௯"
    "০১২৩৪৫৬৭৮৯" "૦૧૨૩૪૫૬૭૮૯" "൦൧൨൩൪൫൬൭൮൯" "੦੧੨੩੪੫੬੭੮੯" "౦౧౨౩౪౫౬౭౮౯",
    "0123456789" * 8,
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def basic(text: str) -> str:
    """NFC, native digits → ASCII, lower-case, strip punctuation, collapse spaces."""
    text = nfc(text).translate(_NATIVE_DIGITS).lower()
    text = _strip_symbols(text, keep="₹./,-")
    return re.sub(r"\s+", " ", text).strip()


def for_wer(text: str) -> str:
    """Normalisation used before WER/CER: like `basic` but also drops . , / -."""
    text = basic(text)
    text = re.sub(r"[.,/\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Romanisation
# --------------------------------------------------------------------------- #

_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "tamil": (0x0B80, 0x0BFF),
}


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    return None


def romanize(text: str) -> str:
    """Transliterate any Devanagari / Kannada / Tamil runs to lower-case ITRANS.

    Latin text is left untouched, so code-mixed output stays readable.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:  # pragma: no cover - dependency is declared in pyproject
        return text

    schemes = {
        "devanagari": sanscript.DEVANAGARI,
        "kannada": sanscript.KANNADA,
        "tamil": sanscript.TAMIL,
    }
    out, run, run_script = [], [], None
    for ch in nfc(text):
        s = _script_of(ch)
        if s != run_script and run:
            chunk = "".join(run)
            out.append(transliterate(chunk, schemes[run_script], sanscript.ITRANS) if run_script else chunk)
            run = []
        run_script = s
        run.append(ch)
    if run:
        chunk = "".join(run)
        out.append(transliterate(chunk, schemes[run_script], sanscript.ITRANS) if run_script else chunk)
    return "".join(out).lower()


# --------------------------------------------------------------------------- #
# Number lexicons
# --------------------------------------------------------------------------- #

SMALL: dict[str, float] = {
    # English
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Hindi (romanised)
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
    "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6, "che": 6, "chah": 6,
    "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "gyaarah": 11, "barah": 12, "baarah": 12, "terah": 13,
    "chaudah": 14, "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18,
    "athaarah": 18, "unnis": 19, "unees": 19, "bees": 20, "bis": 20,
    "pachees": 25, "pachchees": 25, "pachis": 25, "tees": 30, "tis": 30,
    "chalis": 40, "chaalis": 40, "chalees": 40, "pachas": 50, "pachaas": 50,
    "saath": 60, "sattar": 70, "pachattar": 75, "assi": 80, "nabbe": 90, "nabbey": 90,
    "dedh": 1.5, "dhai": 2.5, "dhaai": 2.5,
    # Kannada (romanised)
    "ondu": 1, "eradu": 2, "mooru": 3, "muru": 3, "naalku": 4, "nalku": 4,
    "aidu": 5, "aaru": 6, "elu": 7, "yelu": 7, "entu": 8, "ombattu": 9,
    "hattu": 10, "ippattu": 20, "muvattu": 30, "moovattu": 30, "nalavattu": 40,
    "aivattu": 50, "aravattu": 60, "eppattu": 70, "embattu": 80, "tombattu": 90,
    # Tamil (romanised)
    "onnu": 1, "onru": 1, "rendu": 2, "moonu": 3, "munu": 3, "naalu": 4, "nalu": 4,
    "anju": 5, "ainthu": 5, "ezhu": 7, "yezhu": 7, "ettu": 8, "onbadhu": 9,
    "onbathu": 9, "pathu": 10, "padhu": 10, "irupathu": 20, "muppathu": 30,
    "naarpathu": 40, "narpathu": 40, "aimbathu": 50, "ambathu": 50, "arubathu": 60,
    "ezhubathu": 70, "enbathu": 80, "thonnooru": 90,
    # Devanagari words (common forms)
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "बीस": 20, "पच्चीस": 25, "तीस": 30,
    "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
    "डेढ़": 1.5, "ढाई": 2.5,
    # Hindi 21-31, needed for spoken dates ("ikkis october"). Note "teis" is 23
    # while "tees" is 30: one is a date, the other an amount, and a recogniser
    # confusing them is exactly the kind of error CrER exists to catch.
    "ikkis": 21, "ekkis": 21, "bais": 22, "baees": 22, "teis": 23,
    "chaubis": 24, "chaubees": 24, "chhabbis": 26, "chabbis": 26, "chhabis": 26,
    "sattais": 27, "satayis": 27, "atthais": 28, "athais": 28,
    "untis": 29, "unatis": 29, "untees": 29, "ikattis": 31, "iktis": 31,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "छब्बीस": 26,
    "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "इकतीस": 31,
    # Kannada script. Without these, a Kannada amount is unscoreable the moment
    # a model answers in its own script — which is what Prisma's `verbatim` and
    # `transcribe` formats do.
    "ಸೊನ್ನೆ": 0, "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5,
    "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8, "ಒಂಬತ್ತು": 9, "ಹತ್ತು": 10,
    "ಇಪ್ಪತ್ತು": 20, "ಮೂವತ್ತು": 30, "ನಲವತ್ತು": 40, "ಐವತ್ತು": 50,
    "ಅರವತ್ತು": 60, "ಎಪ್ಪತ್ತು": 70, "ಎಂಬತ್ತು": 80, "ತೊಂಬತ್ತು": 90,
    # Tamil script, both the formal and the spoken forms.
    "ஒன்று": 1, "ஒன்னு": 1, "இரண்டு": 2, "ரெண்டு": 2, "மூன்று": 3, "மூனு": 3,
    "நான்கு": 4, "நாலு": 4, "ஐந்து": 5, "அஞ்சு": 5, "ஆறு": 6, "ஏழு": 7,
    "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10, "இருபது": 20, "முப்பது": 30,
    "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60, "எழுபது": 70, "எண்பது": 80,
    "தொண்ணூறு": 90,
}

HUNDRED = {"hundred", "sau", "nooru", "noor", "सौ", "ನೂರು", "நூறு"}
# Spelling variants matter as much as the words themselves: a recogniser that
# writes "Tis Laak" has understood "tees lakh" perfectly, and scoring that as a
# critical error would blame the model for this lexicon's blind spot. Every
# variant below was observed in real ASR output or is a standard romanisation.
MULTIPLIERS: dict[str, int] = {
    "thousand": 1_000, "k": 1_000, "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000,
    "hajaar": 1_000, "hazaro": 1_000, "hazaron": 1_000,
    "saavira": 1_000, "savira": 1_000, "aayiram": 1_000, "ayiram": 1_000,
    "हज़ार": 1_000, "हजार": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "l": 100_000,
    "laakh": 100_000, "laak": 100_000, "laaks": 100_000, "lakhon": 100_000,
    "laksha": 100_000, "laksham": 100_000, "latcham": 100_000,
    "लाख": 100_000, "लाक": 100_000,  # लाक observed from Whisper on real audio
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000, "karod": 10_000_000,
    "karor": 10_000_000, "karore": 10_000_000, "cror": 10_000_000,
    "karodo": 10_000_000, "karodon": 10_000_000,
    "koti": 10_000_000, "kodi": 10_000_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
    # Kannada and Tamil script units, for the same reason as the digit words.
    "ಸಾವಿರ": 1_000, "ಲಕ್ಷ": 100_000, "ಕೋಟಿ": 10_000_000,
    "ஆயிரம்": 1_000, "லட்சம்": 100_000, "கோடி": 10_000_000,
}
HALF_PREFIX = {"saadhe", "sadhe", "saade", "साढ़े"}
CURRENCY = {"₹", "rs", "rs.", "rupees", "rupaye", "rupay", "rupiya", "rupee", "inr", "रुपये", "रुपए"}

_NUM_TOKEN = re.compile(r"^₹?\d[\d,]*(\.\d+)?$")
_ATTACHED_UNIT = re.compile(r"^₹?(\d[\d,]*(?:\.\d+)?)(k|l|cr|lakh|crore)$")


def _strip_symbols(text: str, keep: str = "₹.,") -> str:
    """Replace punctuation/symbols with spaces while keeping Indic vowel signs.

    Python's `\\w` does not match combining marks (matras), so a regex-based
    strip would split words like "लाख". We filter by Unicode category instead.
    """
    return "".join(
        ch if ch in keep or not unicodedata.category(ch).startswith(("P", "S")) else " "
        for ch in text
    )


def _tokens(text: str) -> list[str]:
    text = nfc(text).translate(_NATIVE_DIGITS).lower()
    text = text.replace("₹", " ₹ ")
    text = _strip_symbols(text)
    return [t.strip(".,") for t in text.split() if t.strip(".,")]


def parse_amounts(text: str) -> list[int]:
    """Return every numeric value in `text`, as integers (rupees for amounts).

    "₹2.25 crore" → 22500000, "tees lakh" → 3000000, "do lakh pachaas hazaar" → 250000,
    "saadhe teen lakh" → 350000, "2,25,00,000" → 22500000.
    """
    tokens = [t for t in _tokens(text) if t not in CURRENCY and t != "₹"]
    values: list[int] = []
    total, current, active, half = 0.0, 0.0, False, False

    def flush() -> None:
        nonlocal total, current, active, half
        if active:
            values.append(round(total + current))
        total, current, active, half = 0.0, 0.0, False, False

    for tok in tokens:
        attached = _ATTACHED_UNIT.match(tok)
        if attached:
            flush()
            num = float(attached.group(1).replace(",", ""))
            values.append(round(num * MULTIPLIERS[attached.group(2)]))
            continue
        if _NUM_TOKEN.match(tok):
            num = float(tok.lstrip("₹").replace(",", ""))
            if active and current and current < 100 and num < 100 and not total:
                flush()  # two bare numbers in a row are separate values
            current += num + (0.5 if half else 0)
            active, half = True, False
            continue
        if tok in HALF_PREFIX:
            half = True
            continue
        if tok in SMALL:
            current += SMALL[tok] + (0.5 if half else 0)
            active, half = True, False
            continue
        if tok in HUNDRED:
            current = (current or 1) * 100
            active = True
            continue
        if tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if active or tok not in {"k", "l"}:
                total += (current or 1) * mult
                current = 0.0
                active = True
            continue
        if tok in {"and", "aur"} and active:
            continue
        flush()
    flush()
    return values


# --------------------------------------------------------------------------- #
# Spoken digit strings
# --------------------------------------------------------------------------- #

DIGIT_WORDS: dict[str, str] = {
    "zero": "0", "oh": "0", "shunya": "0", "sunna": "0", "pojyam": "0", "soonya": "0",
    "one": "1", "ek": "1", "ondu": "1", "onnu": "1",
    "two": "2", "do": "2", "eradu": "2", "rendu": "2",
    "three": "3", "teen": "3", "tin": "3", "mooru": "3", "moonu": "3",
    "four": "4", "char": "4", "chaar": "4", "naalku": "4", "naalu": "4",
    "five": "5", "paanch": "5", "panch": "5", "aidu": "5", "anju": "5",
    "six": "6", "chhe": "6", "che": "6", "aaru": "6",
    "seven": "7", "saat": "7", "sat": "7", "elu": "7", "ezhu": "7",
    "eight": "8", "aath": "8", "entu": "8", "ettu": "8",
    "nine": "9", "nau": "9", "ombattu": "9", "onbadhu": "9",
    # --- Native scripts -------------------------------------------------- #
    # These are not optional. Prisma's `verbatim` and `transcribe` formats emit
    # native script for hi/kn/ta, while `codemix` and `transliterate` emit
    # Latin. With Latin-only digit words, a spoken account number is
    # unscoreable in native script, so the Latin formats would beat the native
    # ones on CrER for reasons that have nothing to do with accuracy — and that
    # comparison is the point of the leaderboard.
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "सात": "7",
    "आठ": "8", "नौ": "9",
    "ಸೊನ್ನೆ": "0", "ಒಂದು": "1", "ಎರಡು": "2", "ಮೂರು": "3", "ನಾಲ್ಕು": "4",
    "ಐದು": "5", "ಆರು": "6", "ಏಳು": "7", "ಎಂಟು": "8", "ಒಂಬತ್ತು": "9",
    "பூஜ்யம்": "0", "சுழி": "0", "ஒன்று": "1", "ஒன்னு": "1",
    "இரண்டு": "2", "ரெண்டு": "2", "மூன்று": "3", "மூனு": "3",
    "நான்கு": "4", "நாலு": "4", "ஐந்து": "5", "அஞ்சு": "5",
    "ஆறு": "6", "ஏழு": "7", "எட்டு": "8", "ஒன்பது": "9",
}
_REPEAT = {"double": 2, "triple": 3}

#: A numeric date is `parse_dates`' job. Without masking it, "account number
#: char char ek saat, 14/11/2026" collapses into one 12-digit run and the
#: read-back agent confirms a number the caller never said.
_DATE_MASK = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
#: A comma between digits is a thousands separator, not a boundary.
_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")
_BREAK = "\x00"  # hard: a masked date, never part of a digit run
_SOFT = "\x01"  # punctuation: a boundary only between multi-digit numbers


def _is_single_digit(token: str) -> bool:
    """A token carrying exactly one digit — a digit word, or one numeral."""
    return token in DIGIT_WORDS or (len(token) == 1 and token.isdigit())


def _digit_segments(text: str) -> list[list[str]]:
    """Split into runs of tokens that may belong to the same digit string.

    Punctuation is treated carefully, because both readings are common:

    * **A list separator.** Speakers and recognisers punctuate dictated digits:
      Prisma returns "ek, do, teen, chaar, paanch, che." for one six-digit
      account number. Breaking there would score a *perfect* transcript as a
      total miss.
    * **A real boundary.** "pin 1234, otp 5678" is two values, and joining them
      would invent a number nobody said.

    The discriminator is what sits on either side: single digits are a dictated
    list, multi-digit numbers are separate values. A masked date is always a
    hard break — it is `parse_dates`' job — which on its own fixes the original
    gluing bug ("...ek saat, 14/11/2026").
    """
    t = nfc(text).translate(_NATIVE_DIGITS).lower()
    t = _DATE_MASK.sub(_BREAK, t)
    t = _GROUP_COMMA.sub("", t)
    marked = "".join(
        ch
        if ch in ("₹", _BREAK) or not unicodedata.category(ch).startswith(("P", "S"))
        else _SOFT
        for ch in t
    )

    segments: list[list[str]] = []
    for hard_chunk in marked.split(_BREAK):
        pieces = [piece.split() for piece in hard_chunk.split(_SOFT)]
        pieces = [p for p in pieces if p]
        if not pieces:
            continue
        merged = [pieces[0]]
        for piece in pieces[1:]:
            previous = merged[-1]
            if _is_single_digit(previous[-1]) and _is_single_digit(piece[0]):
                previous.extend(piece)  # a dictated list, not a boundary
            else:
                merged.append(piece)
        segments.extend(merged)
    return segments


def parse_digit_strings(text: str, min_len: int = 3) -> list[str]:
    """Extract digit sequences, joining spoken digits and grouped numerals.

    "account number char char ek saat" → ["4417"]; "double four 1 7" → ["4417"];
    "4417" and "44 17" → ["4417"]; "pin 1234 aur otp 5678" → ["1234", "5678"].

    Numeric dates are excluded — they are reported by `parse_dates`.
    """
    out: list[str] = []
    for segment in _digit_segments(text):
        buf, repeat = "", 1
        for tok in segment:
            if tok in _REPEAT:
                repeat = _REPEAT[tok]
                continue
            if tok.isdigit():
                buf += tok * (repeat if len(tok) == 1 else 1)
            elif tok in DIGIT_WORDS:
                buf += DIGIT_WORDS[tok] * repeat
            else:
                if len(buf) >= min_len:
                    out.append(buf)
                buf = ""
            repeat = 1
        if len(buf) >= min_len:
            out.append(buf)
    return out


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #

# "ji" / "जी" are deliberately EXCLUDED, like "na" below. They are politeness
# particles, not agreement: "जी, काव्या अय्यर बोल रही हूँ" ("yes, this is Kavya
# Iyer speaking") is someone giving their name, and scoring it as consent
# invents an agreement the caller never gave — the most dangerous possible
# direction for this metric to err. Genuine affirmations in the same breath
# ("haan ji", "ji theek hai") are still caught by "haan" and "theek".
YES = {
    "haan", "han", "haanji", "ha", "haa", "yes", "yeah", "yep", "ok",
    "okay", "theek", "thik", "sahi", "bilkul", "sure", "correct", "right",
    "haudu", "houdu", "howdu", "sari", "aamaa", "aama", "amam", "seri",
    "हाँ", "हां", "ठीक", "सही", "ಹೌದು", "ಸರಿ", "ஆமா", "ஆமாம்", "சரி",
}
# "na" is deliberately excluded: in Hindi it is far more often a tag ("hai na?")
# than a refusal.
NO = {
    "nahi", "nahin", "nai", "no", "nope", "mat", "illa", "illai", "beda",
    "venda", "vendam", "wrong", "galat",
    "नहीं", "नही", "ಇಲ್ಲ", "ಬೇಡ", "இல்ல", "இல்லை", "வேண்டாம்",
}


#: Words naming a thing that *carries* a number. Deliberately narrow: only
#: nouns for numbered objects, never "loan" or "payment", which name money.
IDENTIFIER_CUES = {
    "account", "accountno", "number", "nambar", "numbers", "policy", "order",
    "otp", "pin", "card", "reference", "digits", "khata", "aadhaar", "pan",
    "नंबर", "खाता", "अकाउंट", "कार्ड",
    "ಖಾತೆ", "ನಂಬರ್", "கணக்கு", "நம்பர்",
}

#: Words that mean the number beside them is money. CURRENCY and MULTIPLIERS
#: carry most of it; these are the nouns that introduce a sum without naming
#: a unit, as in "policy ka amount 30,00,000".
MONEY_CUES = {
    "amount", "rakam", "paisa", "paise", "price", "cost", "fee", "fees",
    "charge", "charges", "bill", "salary", "loan", "emi", "premium",
    "balance", "payment", "refund", "recharge", "claim", "sanction",
    "राशि", "रकम", "कीमत", "भुगतान", "लोन", "किस्त",
    "ಮೊತ್ತ", "ಹಣ", "தொகை", "பணம்",
}


def mentions_money(text: str) -> bool:
    """True when any word in `text` says a number is money.

    Weaker than `numbers_are_identifiers`, which asks which cue is nearest.
    This only asks whether money is spoken of at all, which is what decides
    a number that is otherwise explained -- a year inside a date, say.
    """
    for tok in _tokens(text):
        if tok in CURRENCY or tok in MULTIPLIERS or tok in MONEY_CUES:
            return True
        if _ATTACHED_UNIT.match(tok):
            return True
    return False


def _is_number_token(tok: str) -> bool:
    return bool(
        _NUM_TOKEN.match(tok) or _ATTACHED_UNIT.match(tok)
        or tok in SMALL or tok in DIGIT_WORDS or tok in HUNDRED
    )


def numbers_are_identifiers(text: str) -> bool:
    """True when every number in `text` is an identifier and none is money.

    Each number is judged by the nearest cue before it, so one sentence can
    carry an amount and an account number and be read correctly:

    >>> numbers_are_identifiers("account number char char ek saat")
    True
    >>> numbers_are_identifiers("policy ka amount 30,00,000 hai na")
    False
    >>> numbers_are_identifiers("loan teen lakh, account number char char ek saat")
    False
    """
    toks = _tokens(text)
    last_cue: Optional[str] = None
    saw_identifier = False

    for i, tok in enumerate(toks):
        if tok in CURRENCY or tok in MULTIPLIERS or tok in MONEY_CUES:
            last_cue = "money"
            continue
        if tok in IDENTIFIER_CUES:
            last_cue = "identifier"
            continue
        if not _is_number_token(tok):
            continue

        # A unit sitting on or right after the number settles it outright:
        # "2.25cr", "teen lakh", "400 rupaye".
        if _ATTACHED_UNIT.match(tok):
            return False
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if nxt in MULTIPLIERS or nxt in CURRENCY:
            return False

        if last_cue == "money":
            return False
        if last_cue == "identifier":
            saw_identifier = True

    return saw_identifier


def consent_polarity(text: str) -> Optional[str]:
    """'no' if any negative token is present, else 'yes' if any positive token, else None."""
    toks = set(_tokens(text))
    if toks & NO:
        return "no"
    if toks & YES:
        return "yes"
    return None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
         ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
         ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
         ("december", "dec")]
    )
    for m in names
}
#: Month names beyond English. Prisma's `verbatim` and `transcribe` formats
#: return native script, and its romanised formats return spellings like
#: "navambar", so an English-only month table scores a correct date as a miss.
MONTH_VARIANTS: dict[str, int] = {
    # romanised Hindi
    "janvari": 1, "janwari": 1, "farvari": 2, "pharvari": 2, "febrvari": 2,
    "aprail": 4, "mai": 5, "julai": 7, "agast": 8, "sitambar": 9,
    "aktubar": 10, "aktoobar": 10, "navambar": 11, "navmbar": 11, "disambar": 12,
    # Devanagari
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
    "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
    # Kannada
    "ಜನವರಿ": 1, "ಫೆಬ್ರವರಿ": 2, "ಮಾರ್ಚ್": 3, "ಏಪ್ರಿಲ್": 4, "ಮೇ": 5, "ಜೂನ್": 6,
    "ಜುಲೈ": 7, "ಆಗಸ್ಟ್": 8, "ಸೆಪ್ಟೆಂಬರ್": 9, "ಅಕ್ಟೋಬರ್": 10,
    "ನವೆಂಬರ್": 11, "ಡಿಸೆಂಬರ್": 12,
    # Tamil
    "ஜனவரி": 1, "பிப்ரவரி": 2, "மார்ச்": 3, "ஏப்ரல்": 4, "மே": 5, "ஜூன்": 6,
    "ஜூலை": 7, "ஆகஸ்ட்": 8, "செப்டம்பர்": 9, "அக்டோபர்": 10,
    "நவம்பர்": 11, "டிசம்பர்": 12,
}
MONTHS.update(MONTH_VARIANTS)

#: A spoken year is a phrase: "do hazar chabbis" -> 2026. Values outside this
#: window are not a year anyone is stating in a call.
_YEAR_RANGE = (1900, 2100)
_ORDINAL = re.compile(r"(st|nd|rd|th)$")

_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")


def _day_at(tokens: list[str], index: int) -> tuple[Optional[int], int]:
    """Read a day ending at `index`, returning (day, tokens consumed).

    A day is a numeral ("14", "21st") or a number word — one token in Hindi
    ("ikkis"), two in Kannada and Tamil ("ippattu ondu", "irupathu onnu").
    """
    if index < 0:
        return None, 0
    token = _ORDINAL.sub("", tokens[index])
    if token.isdigit() and 1 <= int(token) <= 31:
        return int(token), 1
    single = SMALL.get(token)
    # A compound such as "ippattu ondu" (20 + 1) must be tried BEFORE the bare
    # unit, or "ondu" alone wins and the 21st becomes the 1st.
    if index >= 1 and single is not None and 1 <= single <= 9:
        tens = SMALL.get(tokens[index - 1])
        if tens is not None and 20 <= tens <= 30 and float(tens).is_integer():
            total = int(tens + single)
            if total <= 31:
                return total, 2
    if single is not None and float(single).is_integer() and 1 <= single <= 31:
        return int(single), 1
    return None, 0


def _year_after(tokens: list[str], index: int, default_year: int) -> int:
    """Read a year starting at `index`: "2027", or "do hazar chabbis" -> 2026."""
    if index >= len(tokens):
        return default_year
    if tokens[index].isdigit() and len(tokens[index]) == 4:
        year = int(tokens[index])
        if _YEAR_RANGE[0] <= year <= _YEAR_RANGE[1]:
            return year
    # A spoken year is a phrase; hand the next few tokens to the amount parser.
    for width in (4, 3, 2):
        phrase = " ".join(tokens[index : index + width])
        for value in parse_amounts(phrase):
            if _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1]:
                return value
    return default_year


def parse_dates(text: str, default_year: int = 2026) -> list[str]:
    """ISO dates from dd/mm/yyyy, or a spoken "<day> <month> [year]".

    Handles the forms real recognisers return, not just the numeric one:
    "14/11/2026", "3 December 2026", "ikkis october do hazar chabbis",
    "tees November do hazar chabbis", "8 ஜனவரி 2027".

    Without spoken days, native/romanised month names and spoken years, the
    date column measures whether a provider applies inverse text normalisation
    rather than whether it heard the date — and every non-ITN format scores a
    near-total miss on a correct transcription.
    """
    found: list[str] = []
    t = nfc(text).translate(_NATIVE_DIGITS).lower()
    for d, m, y in _NUMERIC_DATE.findall(t):
        year = int(y) + 2000 if len(y) == 2 else int(y)
        try:
            found.append(date(year, int(m), int(d)).isoformat())
        except ValueError:
            pass

    tokens = _tokens(t)
    for i, token in enumerate(tokens):
        month = MONTHS.get(token)
        if month is None:
            continue
        day, _consumed = _day_at(tokens, i - 1)
        if day is None:
            continue
        year = _year_after(tokens, i + 1, default_year)
        try:
            found.append(date(year, month, day).isoformat())
        except ValueError:
            pass
    return found
