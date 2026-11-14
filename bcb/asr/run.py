"""Run ASR providers over degraded clips and write transcripts.

    python -m bcb.asr.run --provider prisma_v3_transcribe --condition phone_harsh --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from bcb.asr.providers import build_all


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--provider", default="all")
    ap.add_argument("--condition", default="all")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    clips = [json.loads(line) for line in open(paths["clips"])]
    if args.limit:
        clips = clips[: args.limit]
    conditions = list(cfg["conditions"]) if args.condition == "all" else args.condition.split(",")

    for name, provider in build_all(cfg, args.provider).items():
        for cond in conditions:
            out_path = Path(paths["transcripts"]) / name / f"{cond}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a sibling temp file and rename only on success. Opening
            # the real path with "w" truncates it the instant the run starts,
            # so a long or interrupted run leaves the PREVIOUS good transcripts
            # destroyed and `make score` silently scores without that condition
            # — producing different numbers from the same tree, with no warning.
            tmp_path = out_path.with_suffix(".jsonl.partial")
            n_err = n_new = 0
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for clip in tqdm(clips, desc=f"{name}/{cond}"):
                        wav = (Path(paths["audio"]) / cond / clip["voice_source"]
                               / f"{clip['clip_id']}.wav")
                        res = provider.transcribe(str(wav), clip["lang"])
                        n_err += res["error"] is not None
                        n_new += not res["cached"]
                        f.write(json.dumps({
                            "clip_id": clip["clip_id"], "voice_source": clip["voice_source"],
                            "condition": cond, "provider": name, "text": res["text"],
                            "latency_ms": res["latency_ms"], "error": res["error"],
                        }, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(out_path)
            except BaseException:
                # Keep the partial file for inspection; the API responses are
                # cached, so resuming costs nothing.
                print(f"\n{name}/{cond}: interrupted — previous {out_path.name} left intact, "
                      f"partial output in {tmp_path.name}")
                raise
            print(f"{name}/{cond}: {len(clips)} clips, {n_new} new API calls, {n_err} errors")


if __name__ == "__main__":
    main()
