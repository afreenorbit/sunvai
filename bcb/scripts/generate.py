"""Generate call scripts with an LLM, validate them, and assign splits.

    python -m bcb.scripts.generate --config configs/bench.yaml --out data/scripts/raw.jsonl

Uses LiteLLM so any provider works (model from SCRIPT_LLM_MODEL). Invalid
scripts (bad JSON, span surface not in text, unparseable values) are retried.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm import tqdm

from bcb.schema import CallScript
from bcb.scripts.prompts import DOMAIN_GUIDE, LANG_GUIDE, SYSTEM, USER_TEMPLATE

MAX_RETRIES = 3


def generate_one(call_id: str, lang: str, domain: str, n_turns: int, model: str) -> CallScript:
    import litellm  # pip install -e ".[scripts]"

    prompt = USER_TEMPLATE.format(
        lang_guide=LANG_GUIDE[lang], domain_guide=DOMAIN_GUIDE[domain],
        call_id=call_id, n_turns=n_turns, lang=lang, domain=domain,
    )
    last_err = None
    for _ in range(MAX_RETRIES):
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            data["call_id"], data["lang"], data["domain"] = call_id, lang, domain
            return CallScript.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = exc
    raise RuntimeError(f"{call_id}: failed after {MAX_RETRIES} tries: {last_err}")


def assign_splits(scripts: list[CallScript], ratios: dict) -> None:
    """Split by call, stratified by language, deterministic by seed."""
    rng = random.Random(ratios.get("seed", 42))
    by_lang: dict[str, list[CallScript]] = {}
    for s in scripts:
        by_lang.setdefault(s.lang, []).append(s)
    for group in by_lang.values():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        for i, s in enumerate(group):
            s.split = "train" if i < n_train else "val" if i < n_train + n_val else "test"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--out", default="data/scripts/raw.jsonl")
    ap.add_argument("--per-lang", type=int, default=None, help="override calls_per_language")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = os.getenv("SCRIPT_LLM_MODEL", "openai/gpt-4o-mini")
    per_lang = args.per_lang or cfg["calls_per_language"]
    domains = cfg["domains"]

    jobs = []
    for lang in (entry["code"] for entry in cfg["languages"]):
        for i in range(per_lang):
            domain = domains[i % len(domains)]
            jobs.append((f"{lang[:2]}_{domain}_{i:03d}", lang, domain))

    scripts, failures = [], []
    for call_id, lang, domain in tqdm(jobs):
        try:
            scripts.append(generate_one(call_id, lang, domain, cfg["customer_turns_per_call"], model))
        except RuntimeError as exc:
            failures.append(str(exc))

    assign_splits(scripts, cfg["split"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in scripts:
            f.write(s.model_dump_json() + "\n")
    print(f"wrote {len(scripts)} scripts to {args.out}; {len(failures)} failures")
    for msg in failures:
        print("  -", msg)


if __name__ == "__main__":
    main()
