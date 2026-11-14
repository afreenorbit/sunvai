"""Telephony degradation: band-limit, resample, codec, noise, packet loss.

Each condition in configs/bench.yaml is a chain of these steps. Everything is
deterministic given a seed, so a clip's degraded audio can be rebuilt exactly.

    python -m bcb.audio.degrade --config configs/bench.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, resample_poly, sosfiltfilt

# --------------------------------------------------------------------------- #
# Primitive steps
# --------------------------------------------------------------------------- #


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def telephone_band(x: np.ndarray, sr: int, low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    """Band-limit to the classic 300–3400 Hz telephone channel."""
    high = min(high, sr / 2 - 100)
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def mulaw(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """G.711 μ-law round trip (8-bit companding and back)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # 8-bit quantisation
    return (np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)).astype(np.float32)


def gsm(x: np.ndarray, sr: int) -> np.ndarray:
    """GSM 06.10 round trip via sox (requires sox built with GSM support)."""
    if not shutil.which("sox"):
        raise RuntimeError("sox not found; install sox or use codec: mulaw")
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d, "in.wav"), Path(d, "x.gsm"), Path(d, "out.wav")
        sf.write(src, x, sr)
        subprocess.run(["sox", str(src), "-r", "8000", "-c", "1", str(enc)], check=True)
        subprocess.run(["sox", str(enc), "-r", str(sr), str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float32")
    return y


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix a random segment of `noise` into `x` at the given SNR."""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = rng.integers(0, len(noise) - len(x) + 1)
    n = noise[start : start + len(x)]
    p_sig = np.mean(x**2) + 1e-12
    p_noise = np.mean(n**2) + 1e-12
    n = n * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    y = x + n
    peak = np.max(np.abs(y))
    return (y / peak * 0.99 if peak > 1 else y).astype(np.float32)


def packet_loss(
    x: np.ndarray, sr: int, rate: float, burst: float, rng: np.random.Generator, frame_ms: int = 20
) -> np.ndarray:
    """Drop 20 ms frames with a two-state (Gilbert) model.

    `rate` is the long-run fraction of lost frames; `burst` the mean burst length.
    Lost frames are zeroed (no concealment), the worst case for a recogniser.
    """
    if rate <= 0:
        return x
    frame = int(sr * frame_ms / 1000)
    n_frames = int(np.ceil(len(x) / frame))
    p_bad_to_good = 1.0 / max(burst, 1.0)
    p_good_to_bad = rate * p_bad_to_good / max(1e-9, 1 - rate)
    y = x.copy()
    bad = False
    for i in range(n_frames):
        bad = rng.random() < (1 - p_bad_to_good if bad else p_good_to_bad)
        if bad:
            y[i * frame : (i + 1) * frame] = 0.0
    return y


# --------------------------------------------------------------------------- #
# Condition chain
# --------------------------------------------------------------------------- #


def apply_condition(
    x: np.ndarray,
    sr: int,
    cond: dict,
    noise_bank: Optional[list[np.ndarray]] = None,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Apply one condition (a dict from bench.yaml) to mono float32 audio."""
    rng = np.random.default_rng(seed)
    x = to_mono(x).astype(np.float32)
    if not cond:
        return x, sr

    target_sr = cond.get("sample_rate", sr)
    if target_sr <= 8000:
        x = telephone_band(x, sr)
    x = resample(x, sr, target_sr)
    sr = target_sr

    if "noise_snr_db" in cond:
        if not noise_bank:
            raise ValueError("condition needs noise but the noise bank is empty (see data/noise/README.md)")
        noise = noise_bank[rng.integers(len(noise_bank))]
        x = add_noise(x, noise, cond["noise_snr_db"], rng)

    codec = cond.get("codec", "none")
    if codec == "mulaw":
        x = mulaw(x)
    elif codec == "gsm":
        x = gsm(x, sr)

    if cond.get("packet_loss"):
        x = packet_loss(x, sr, cond["packet_loss"], cond.get("loss_burst", 1), rng)
    return x, sr


def load_noise_bank(noise_dir: Path, sr: int) -> list[np.ndarray]:
    bank = []
    for p in sorted(noise_dir.glob("*.wav")):
        n, nsr = sf.read(p, dtype="float32")
        bank.append(resample(to_mono(n), nsr, sr))
    return bank


def clip_seed(clip_id: str, condition: str) -> int:
    return int(hashlib.sha1(f"{clip_id}|{condition}".encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bench.yaml")
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    audio_root = Path(paths["audio"])
    clips = [json.loads(line) for line in open(paths["clips"])]
    conditions = cfg["conditions"]
    if args.only:
        conditions = {k: v for k, v in conditions.items() if k in args.only.split(",")}

    banks: dict[int, list[np.ndarray]] = {}
    for name, cond in conditions.items():
        if name == "clean":
            continue
        made = 0
        for clip in clips:
            src = audio_root / "clean" / clip["voice_source"] / f"{clip['clip_id']}.wav"
            dst = audio_root / name / clip["voice_source"] / f"{clip['clip_id']}.wav"
            if dst.exists():
                continue
            x, sr = sf.read(src, dtype="float32")
            tsr = cond.get("sample_rate", sr)
            if "noise_snr_db" in cond and tsr not in banks:
                banks[tsr] = load_noise_bank(Path(paths["noise"]), tsr)
            y, ysr = apply_condition(x, sr, cond, banks.get(tsr), clip_seed(clip["clip_id"], name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, y, ysr, subtype="PCM_16")
            made += 1
        print(f"{name}: wrote {made} clips")


if __name__ == "__main__":
    main()
