"""Generate a synthetic noise bank so the pipeline runs with no downloads.

    python -m bcb.audio.noise --out data/noise

**These are not recordings.** They are procedurally generated approximations of
the backgrounds an Indian support call actually sits in — a ceiling fan, street
traffic, a TV in the next room, a market crowd, a pressure cooker. They exist
so that `make nokey` is a genuinely one-command reproducible path, and so CI
can exercise the noisy conditions without shipping licensed audio.

Real recordings are better and take precedence: any `.wav` you drop into
`data/noise/` is loaded alongside these, and files named `synthetic_*.wav` are
skipped once you pass `--real-only` to the degrader. Record the source and
licence of every real file in `data/noise/SOURCES.csv`.

What makes these usable rather than decorative: speech is masked by energy in
its own band, so each generator shapes noise into the 100–3500 Hz telephone
range, and the crowd-like ones are amplitude-modulated at 2–8 Hz, the syllable
rate that makes babble the hardest masker for a recogniser. A flat white hiss
would be both unrealistic and far too easy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

SR = 16_000
DURATION = 20.0


def _bandpass(x: np.ndarray, low: float, high: float, sr: int = SR, order: int = 4) -> np.ndarray:
    high = min(high, sr / 2 - 100)
    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise — the spectral shape most natural backgrounds have."""
    white = rng.standard_normal(n)
    # Voss-style single-pole cascade; cheap and close enough to 1/f.
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white).astype(np.float32)


def _modulate(x: np.ndarray, rate_hz: float, depth: float, rng: np.random.Generator) -> np.ndarray:
    """Amplitude-modulate at a syllable-like rate, with a random phase."""
    t = np.arange(len(x)) / SR
    env = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t + rng.uniform(0, 2 * np.pi)))
    return (x * env).astype(np.float32)


def _normalise(x: np.ndarray, peak: float = 0.7) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #


def ceiling_fan(n: int, rng: np.random.Generator) -> np.ndarray:
    """Low rumble plus a blade-rate thrum: the default Indian indoor background."""
    t = np.arange(n) / SR
    rumble = _bandpass(_pink(n, rng), 60, 600)
    blade_hz = rng.uniform(4.0, 6.5)  # ~250–400 rpm
    thrum = 0.35 * np.sin(2 * np.pi * blade_hz * t)
    motor = 0.12 * np.sin(2 * np.pi * rng.uniform(95, 115) * t)
    return _normalise(rumble * (1 + thrum) + motor)


def street_traffic(n: int, rng: np.random.Generator) -> np.ndarray:
    """Engine rumble, tyre hiss and the horns you cannot avoid."""
    base = _bandpass(_pink(n, rng), 80, 2500)
    out = base.copy()
    t = np.arange(n) / SR
    for _ in range(rng.integers(4, 9)):  # horns
        start = int(rng.uniform(0, max(1, n - SR)))
        length = int(rng.uniform(0.18, 0.5) * SR)
        length = min(length, n - start)
        if length <= 0:
            continue
        f0 = rng.uniform(380, 520)
        local = t[:length]
        horn = sum(
            (0.6 / k) * np.sin(2 * np.pi * f0 * k * local) for k in (1, 2, 3)
        )
        env = np.minimum(1.0, np.linspace(0, 6, length)) * np.linspace(1.0, 0.25, length)
        out[start : start + length] += 0.5 * horn * env
    return _normalise(out)


def tv_next_room(n: int, rng: np.random.Generator) -> np.ndarray:
    """A serial through a wall: speech-shaped, band-limited, music stings."""
    speechy = _modulate(_bandpass(_pink(n, rng), 200, 3000), rng.uniform(3.5, 5.0), 0.7, rng)
    # The wall rolls off the top, but a thin partition still passes most of the
    # F1/F2 region — which is exactly why a TV next door masks speech. The low
    # cut is steep on purpose: without it the 1/f tilt puts most of the energy
    # under 300 Hz and this becomes rumble rather than muffled speech.
    through_wall = _normalise(_bandpass(speechy, 320, 2600, order=6), 1.0)
    t = np.arange(n) / SR
    # A music bed under the dialogue. Scaled against the *normalised* speech
    # band: a fixed amplitude would dominate once the steep filter above has
    # attenuated the noise, turning this clip back into low-frequency rumble.
    sting = 0.10 * np.sin(2 * np.pi * rng.uniform(210, 340) * t) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    )
    return _normalise(through_wall + sting)


def market_crowd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Many overlapping voices — the hardest masker there is."""
    out = np.zeros(n, dtype=np.float32)
    for _ in range(8):  # eight "talkers" at different syllable rates
        voice = _bandpass(_pink(n, rng), rng.uniform(180, 320), rng.uniform(2400, 3400))
        out += _modulate(voice, rng.uniform(2.0, 7.0), rng.uniform(0.5, 0.9), rng)
    return _normalise(out)


def kitchen(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pressure-cooker whistle and utensil clatter."""
    out = _bandpass(_pink(n, rng), 200, 3000) * 0.4
    t = np.arange(n) / SR
    for _ in range(rng.integers(2, 5)):  # whistles
        start = int(rng.uniform(0, max(1, n - 2 * SR)))
        length = min(int(rng.uniform(0.8, 1.8) * SR), n - start)
        if length <= 0:
            continue
        f0 = rng.uniform(1900, 2600)
        local = t[:length]
        warble = f0 + 40 * np.sin(2 * np.pi * 7 * local)
        whistle = np.sin(2 * np.pi * np.cumsum(warble) / SR)
        env = np.minimum(1.0, np.linspace(0, 8, length)) * np.minimum(
            1.0, np.linspace(8, 0, length)
        )
        out[start : start + length] += 0.55 * whistle * env
    for _ in range(rng.integers(6, 14)):  # clatter
        start = int(rng.uniform(0, max(1, n - SR // 4)))
        length = min(int(rng.uniform(0.02, 0.08) * SR), n - start)
        if length <= 0:
            continue
        click = _bandpass(rng.standard_normal(length).astype(np.float32), 1800, 6000)
        out[start : start + length] += 0.5 * click * np.linspace(1, 0, length)
    return _normalise(out)


def railway_platform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Crowd plus a tannoy announcement rolling underneath."""
    crowd = market_crowd(n, rng) * 0.7
    tannoy = _bandpass(
        _modulate(_pink(n, rng), rng.uniform(2.5, 4.0), 0.85, rng), 400, 2200
    )
    t = np.arange(n) / SR
    duck = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t)  # announcements come and go
    return _normalise(crowd + 0.9 * tannoy * duck)


GENERATORS = {
    "ceiling_fan": ceiling_fan,
    "street_traffic": street_traffic,
    "tv_next_room": tv_next_room,
    "market_crowd": market_crowd,
    "kitchen": kitchen,
    "railway_platform": railway_platform,
}

#: Written alongside the wavs so provenance is recorded, as for real clips.
PREFIX = "synthetic_"


def generate(out_dir: Path, seconds: float = DURATION, seed: int = 42,
             variants: int = 2) -> list[Path]:
    """Write the bank. Deterministic: the same seed gives byte-identical files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SR)
    written: list[Path] = []
    rows = []
    for name, fn in GENERATORS.items():
        for v in range(variants):
            # hashlib, not hash(): Python salts str hashing per process, so
            # hash() would give different audio on every interpreter run.
            digest = hashlib.sha1(f"{name}|{v}|{seed}".encode()).hexdigest()[:8]
            rng = np.random.default_rng(int(digest, 16))
            audio = fn(n, rng)
            path = out_dir / f"{PREFIX}{name}_{v}.wav"
            sf.write(path, audio, SR, subtype="PCM_16")
            written.append(path)
            rows.append({
                "file": path.name,
                "source": "generated by bcb.audio.noise",
                "licence": "CC0 (procedurally generated, not a recording)",
                "notes": f"{name}, variant {v}, seed {seed}",
            })

    manifest = out_dir / "SOURCES_synthetic.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "licence", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/noise")
    ap.add_argument("--seconds", type=float, default=DURATION)
    ap.add_argument("--variants", type=int, default=2, help="clips per category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    written = generate(Path(args.out), args.seconds, args.seed, args.variants)
    print(f"wrote {len(written)} synthetic noise clips to {args.out}")
    print("These are generated, not recorded. Drop real recordings into the same")
    print("directory for a stronger benchmark — see data/noise/README.md.")


if __name__ == "__main__":
    main()
