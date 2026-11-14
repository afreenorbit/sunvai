"""Streaming latency: first-word and final latency per provider.

    python -m bcb.eval.latency --provider prisma_v3_stream --n 50
    python -m bcb.eval.latency --provider prisma_v3_stream --condition phone_noise

Streams raw PCM over the provider's WebSocket at real-time cadence and records:

  first_word_ms = first non-empty partial transcript - first audio frame sent
  final_ms      = final transcript                   - last audio frame sent

Both are measured against *sent* audio, so they are comparable across clips of
different lengths. Audio is paced in real time: sending faster would measure
the server's throughput rather than the latency a caller hears.

The frame size is dictated by each provider's protocol (Prisma requires exactly
1024-byte frames), so the client declares it and this loop follows.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Optional

import numpy as np
import soundfile as sf
import yaml
from dotenv import load_dotenv

from bcb.asr.stream import build_stream
from bcb.audio.degrade import resample, to_mono


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, p))


async def measure(client, wav: str, lang: str, timeout: float = 60.0) -> dict:
    """Stream one clip and return its latencies (None where no message arrived)."""
    x, sr = sf.read(wav, dtype="float32")
    x = to_mono(x)
    if sr != client.sample_rate:
        x = resample(x, sr, client.sample_rate)
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()

    samples_per_frame = client.frame_bytes // 2
    frame_seconds = samples_per_frame / client.sample_rate

    ws = await client.connect(lang)
    first_sent: Optional[float] = None
    last_sent: Optional[float] = None
    first_word: Optional[float] = None
    final: Optional[float] = None
    partials = 0
    text = ""

    async def reader() -> None:
        nonlocal first_word, final, partials, text
        async for msg in ws:
            chunk, is_final = client.parse_message(msg)
            now = time.perf_counter()
            if chunk:
                partials += 1
                text = chunk
                if first_word is None:
                    first_word = now
            if is_final:
                final = now
                return

    task = asyncio.create_task(reader())
    try:
        for offset in range(0, len(pcm), client.frame_bytes):
            frame = pcm[offset : offset + client.frame_bytes]
            if first_sent is None:
                first_sent = time.perf_counter()
            await client.send_frame(ws, frame)
            last_sent = time.perf_counter()
            await asyncio.sleep(frame_seconds)
        await client.end_stream(ws)
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            task.cancel()
    finally:
        await ws.close()

    return {
        "first_word_ms": (first_word - first_sent) * 1000 if first_word and first_sent else None,
        "final_ms": (final - last_sent) * 1000 if final and last_sent else None,
        "audio_ms": len(pcm) / 2 / client.sample_rate * 1000,
        "partials": partials,
        "text": text,
    }


def summarise(values: list[Optional[float]]) -> dict:
    v = [x for x in values if x is not None]
    return {
        "n": len(v),
        "p50": float(median(v)) if v else None,
        "p95": _percentile(v, 95),
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--provider", required=True, help="a provider name from configs/bench.yaml")
    ap.add_argument("--condition", default="clean", help="which degraded audio to stream")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="results/latency.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.provider not in cfg["asr"]:
        raise SystemExit(f"{args.provider} not in config. Available: {sorted(cfg['asr'])}")
    spec = cfg["asr"][args.provider]
    if "stream" not in spec:
        raise SystemExit(
            f"{args.provider} has no `stream:` client. Streaming providers in this config: "
            f"{sorted(n for n, s in cfg['asr'].items() if 'stream' in s)}"
        )
    client = build_stream(spec)

    audio_root = Path(cfg["paths"]["audio"]) / args.condition
    clips = [json.loads(line) for line in open(cfg["paths"]["clips"], encoding="utf-8") if line.strip()]
    clips = clips[: args.n]

    rows, failures = [], 0
    for c in clips:
        wav = audio_root / c["voice_source"] / f"{c['clip_id']}.wav"
        if not wav.exists():
            failures += 1
            continue
        try:
            rows.append(asyncio.run(measure(client, str(wav), c["lang"])))
        except Exception as exc:
            failures += 1
            print(f"  ! {c['clip_id']}: {type(exc).__name__}: {exc}")

    if not rows:
        raise SystemExit("no clips measured")

    fw = summarise([r["first_word_ms"] for r in rows])
    fi = summarise([r["final_ms"] for r in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["provider", "condition", "n_clips", "n_failed",
                        "first_word_p50_ms", "first_word_p95_ms",
                        "final_p50_ms", "final_p95_ms"])
        w.writerow([args.provider, args.condition, len(rows), failures,
                    fw["p50"], fw["p95"], fi["p50"], fi["p95"]])
    print(f"{args.provider}/{args.condition}: {len(rows)} clips, {failures} failed")
    print(f"  first word  p50 {fw['p50']:.0f} ms  p95 {fw['p95']:.0f} ms" if fw["p50"]
          else "  first word  no partials received")
    print(f"  final       p50 {fi['p50']:.0f} ms  p95 {fi['p95']:.0f} ms" if fi["p50"]
          else "  final       no final transcript received")


if __name__ == "__main__":
    main()
