"""Render every customer turn with each configured TTS voice source.

    python -m bcb.tts.synthesize --config configs/bench.yaml --scripts data/scripts/curated.jsonl

Writes data/audio/clean/{voice_source}/{clip_id}.wav (16 kHz mono) and
data/audio/clips.jsonl. Human recordings dropped into
data/audio/clean/human/{clip_id}.wav are picked up automatically.

Two independent engines render every turn, so no result can be an artefact of
one synthesiser: Gnani **Timbre** and Microsoft **Edge** neural voices. Timbre
returns a 16 kHz wav directly; Edge returns mp3 and needs ffmpeg.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from bcb.schema import CallScript
from bcb.tts import timbre

EDGE_VOICES = {
    "hi-IN": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "kn-IN": ["kn-IN-SapnaNeural", "kn-IN-GaganNeural"],
    "ta-IN": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
}


def to_wav16k(src: Path, dst: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert Edge TTS output to wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def gnani_timbre(
    text: str,
    lang: str,
    dst: Path,
    voice: Optional[str] = None,
    model: str = "timbre-v3.0",
) -> str:
    """Synthesise one line with Gnani Timbre. Returns the voice name used."""
    return timbre.synthesize_to_file(text, lang, Path(dst), voice=voice, model=model)


def edge_tts(text: str, lang: str, dst: Path, voice: Optional[str] = None) -> str:
    import edge_tts as et

    pool = EDGE_VOICES[lang]
    voice = voice or pool[hash(Path(dst).stem) % len(pool)]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    try:
        asyncio.run(et.Communicate(text, voice).save(path))
        to_wav16k(Path(path), Path(dst))
    finally:
        Path(path).unlink(missing_ok=True)
    return voice


ENGINES = {"gnani_timbre": gnani_timbre, "edge_tts": edge_tts}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--scripts", default=None, help="defaults to paths.scripts in the config")
    ap.add_argument("--only", default=None, help="comma-separated voice source names")
    ap.add_argument("--limit", type=int, default=0, help="0 = every script")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    audio_root = Path(cfg["paths"]["audio"])
    scripts_path = args.scripts or cfg["paths"]["scripts"]
    scripts = [CallScript.model_validate_json(line)
               for line in open(scripts_path, encoding="utf-8") if line.strip()]
    if args.limit:
        scripts = scripts[: args.limit]

    sources = cfg["tts"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sources = [v for v in sources if v["name"] in wanted]

    clips: list[dict] = []
    failures = 0
    for source in sources:
        engine = ENGINES[source["provider"]]
        kwargs = {k: v for k, v in source.items() if k in ("voice", "model")}
        for s in tqdm(scripts, desc=source["name"]):
            for t in s.customer_turns():
                clip_id = f"{s.call_id}_t{t.index}"
                dst = audio_root / "clean" / source["name"] / f"{clip_id}.wav"
                voice = None
                if dst.exists():
                    voice = "cached"
                else:
                    try:
                        voice = engine(t.text, s.lang, dst, **kwargs)
                    except Exception as exc:
                        failures += 1
                        print(f"  ! {clip_id} [{source['name']}]: {type(exc).__name__}: {exc}")
                        continue
                clips.append({
                    "clip_id": clip_id, "call_id": s.call_id, "turn_index": t.index,
                    "lang": s.lang, "voice_source": source["name"], "voice": voice,
                    "path": str(dst).replace("\\", "/"),
                })

    turns = {f"{s.call_id}_t{t.index}": (s, t) for s in scripts for t in s.customer_turns()}
    human_dir = audio_root / "clean" / "human"
    for wav in sorted(human_dir.glob("*.wav")) if human_dir.exists() else []:
        if wav.stem in turns:
            s, t = turns[wav.stem]
            clips.append({
                "clip_id": wav.stem, "call_id": s.call_id, "turn_index": t.index,
                "lang": s.lang, "voice_source": "human", "voice": "human",
                "path": str(wav).replace("\\", "/"),
            })

    out = Path(cfg["paths"]["clips"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(clips)} clips listed in {out}" + (f"  ({failures} failed)" if failures else ""))


if __name__ == "__main__":
    main()
