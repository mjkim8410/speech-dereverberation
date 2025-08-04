#!/usr/bin/env python3
"""dnsmos_prune_and_norm.py — **fast WAV-in / WAV-out** zero-argument normaliser

* Uses **torchaudio.load** for snappy PCM reads (no MP3 decode any more).
* Skips resample when the file is already at `SR`.
* Computes a single gain so *all* channels stay phase-coherent.
* Writes 16-bit PCM WAV via `soundfile` (pure-Python wheels, no FFmpeg).

Run it with:

    python dnsmos_prune_and_norm.py

Dependencies
------------
    pip install torchaudio soundfile numpy torch tqdm
"""
from __future__ import annotations

import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION – edit and forget about argv/argparse
# ──────────────────────────────────────────────────────────────────────────────
INPUT_GLOB   = "../../data/clean_chunks_wav/*.wav"   # which WAVs to process
OUTPUT_DIR   = "../../data/clean_chunks_normalised/"  # None → next to source
SUFFIX       = "_norm"                                # only if OUTPUT_DIR is None
TARGET_PEAK  = 0.999                                  # absolute peak after gain
DELETE_CLIPPED = True                                 # remove files whose |sample|>1
SR           = 16000                                  # analysis (and output) rate
# ──────────────────────────────────────────────────────────────────────────────

def ensure_outdir(path: Path | None) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def load_wave(path: Path) -> torch.Tensor:
    """Load a WAV, resample to SR if needed (C x T, float32, -1..1)."""
    wav, _ = torchaudio.load(path)  # dtype=float32 already
    if wav.dtype != torch.float32:
        wav = wav.float() / (2 ** 15)  # just in case
    return wav  # (channels, samples)


def peak_gain(wav: torch.Tensor) -> tuple[float, torch.Tensor]:
    peak = wav.abs().max().item()
    if peak <= 0 or peak >= TARGET_PEAK:
        return 1.0, wav
    gain = TARGET_PEAK / peak
    return gain, torch.clamp(wav * gain, -1.0, 1.0)


def save_wav(path: Path, wav: torch.Tensor) -> None:
    """Write CxT float tensor to 16-bit WAV (soundfile expects TxC)."""
    data = wav.t().numpy()  # → (samples, channels)
    sf.write(path, data, SR, subtype="PCM_16")


def main() -> None:
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"No files matched: {INPUT_GLOB}")
        return

    outdir = Path(OUTPUT_DIR) if OUTPUT_DIR else None
    ensure_outdir(outdir)

    deleted = normalised = errors = skipped = 0
    pbar = tqdm(files, desc="Prune + normalise", unit="file")

    for fp in pbar:
        path = Path(fp)
        try:
            wav = load_wave(path)
            peak = wav.abs().max().item()

            # Delete clipped
            if peak > 1.0 and DELETE_CLIPPED:
                os.remove(path)
                deleted += 1
                pbar.set_postfix(dl=deleted, norm=normalised, err=errors, clip=f"{peak:.3f}")
                continue

            gain, wav_scaled = peak_gain(wav)
            if abs(gain - 1.0) < 0.001:  # <0.01 dB → skip
                skipped += 1
                pbar.set_postfix(dl=deleted, norm=normalised, err=errors, skip=skipped)
                continue

            dst = (
                outdir / path.with_suffix(".wav").name if outdir else path.with_name(f"{path.stem}{SUFFIX}.wav")
            )
            save_wav(dst, wav_scaled)
            normalised += 1
            gain_db = 20 * math.log10(gain)
            pbar.set_postfix(dl=deleted, norm=normalised, err=errors, gain=f"{gain_db:+.1f} dB")

        except Exception as e:
            errors += 1
            pbar.set_postfix(dl=deleted, norm=normalised, err=errors, msg=type(e).__name__)

    print(
        f"\nFinished. Deleted {deleted}, normalised {normalised}, skipped {skipped}. Errors: {errors}")


if __name__ == "__main__":
    main()
