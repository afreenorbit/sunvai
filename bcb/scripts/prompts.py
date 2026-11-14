"""Prompt used to write call scripts. Freeze this after the Day 1 human review."""

LANG_GUIDE = {
    "hi-IN": "Hinglish: romanised Hindi mixed naturally with English, as urban and semi-urban callers speak.",
    "kn-IN": "Kanglish: romanised Kannada mixed naturally with English, as Bengaluru and Mysuru callers speak.",
    "ta-IN": "Tanglish: romanised Tamil mixed naturally with English, as Chennai and Coimbatore callers speak.",
}

DOMAIN_GUIDE = {
    "loan": "a customer talking to a lender about a loan amount, EMI or sanction",
    "insurance": "a customer filing or following up an insurance claim",
    "delivery": "a customer calling about a food or parcel delivery problem or refund",
    "kyc": "a customer completing a bank KYC or account verification call",
    "recharge": "a customer recharging a mobile or DTH plan, or disputing a charge",
}

SYSTEM = """You write realistic Indian customer-service phone-call scripts for a speech-recognition benchmark.
Every name, amount, number and date you write is invented. Never write anything resembling a real
Aadhaar, PAN, card, bank account or phone number: digit strings are at most 6 digits."""

USER_TEMPLATE = """Write one phone call in {lang_guide}
Scenario: {domain_guide}.
Call id: {call_id}

Rules:
- Exactly {n_turns} CUSTOMER turns, alternating with short AGENT turns. Customer turns are 6–25 words.
- Every customer turn contains at least one critical span. Across the call, use all of:
  amount (spoken Indian style: "tees lakh", "dedh crore", "pachees hazaar", "do lakh pachaas hazaar"),
  digits (3–6 digits, spoken one by one or as "double four"),
  consent (a clear yes or no in the call's language), name, and date (spoken, e.g. "14 November").
- Mix English words in naturally (loan, EMI, account, refund, OTP, policy, order) — do not over-do it.
- Include at least one near-miss trap per call where mishearing changes meaning
  (e.g. "tees" vs "teen", "pachaas" vs "pachees", "nahi" right after "haan").
- Write everything in Latin script.

Return JSON only, matching:
{{"call_id": str, "lang": "{lang}", "domain": "{domain}",
  "turns": [{{"index": int, "speaker": "customer"|"agent", "text": str,
             "spans": [{{"type": "amount"|"digits"|"consent"|"name"|"date",
                        "surface": exact substring of text,
                        "value": int rupees | digit string | "yes"|"no" | romanised name | "YYYY-MM-DD"}}]}}]}}
Agent turns have "spans": []. Dates are in 2026."""
