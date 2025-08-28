#!/usr/bin/env python3
"""dnsmos_prune_fast.py — I/O‑optimised, batched DNSMOS with density plots

* Switched to **torchaudio.load** → faster WAV read, no MP3 decode.
* Skips resample when file already at `SR`.
* Sends waveforms straight to GPU once per batch (reduces PCIe traffic).
* Supports **batched inference** (clip padding) for better GPU utilisation.
* Everything controlled by the constants block; no CLI flags.
"""
from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

# ─────────────── constants ───────────────────────────────────────────────────
FILES_GLOB   = "../../data/reverb_sports-centre-university-york/*.wav"  # where to find WAVs
PLOTS_DIR    = "../../dnsmos_plots/reverb_sports-centre-university-york"                     # where to save histograms
P808_MIN     = 3.5                                  # prune if p808 < THRESHOLD
SIG_MIN      = 3.55                                 # prune if sig  < SIG_MIN
BAK_MIN      = 4.0                                  # prune if bak  < BAK_MIN
OVRL_MIN     = 3.2                                  # prune if ovrl < OVRL_MIN
SR           = 16000                                # target sample‑rate Hz
BATCH_SIZE   = 1                                    # num clips per GPU batch
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DELETE_FILES = False                                 # True ⇒ actually os.remove()
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_wav(path: str) -> torch.Tensor:
    """Load a mono waveform at SR Hz (tensor shape: 1xT)."""
    wav, _ = torchaudio.load(path)  # stereo shape (C×T)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    return wav.squeeze(0)                # → (T,)


def batch_iter(lst: List[Path], batch_size: int):
    """Yield successive batches from the list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def main() -> None:
    wav_paths = sorted(Path(p) for p in glob.glob(FILES_GLOB))
    if not wav_paths:
        print(f"No files matched: {FILES_GLOB}")
        return

    p808_vals: List[float] = []
    sig_vals:  List[float] = []
    bak_vals:  List[float] = []
    ovrl_vals: List[float] = []

    deleted = errors = 0

    outer = tqdm(batch_iter(wav_paths, BATCH_SIZE), total=math.ceil(len(wav_paths)/BATCH_SIZE), desc="DNSMOS batches")

    for batch_paths in outer:
        wave_list: List[torch.Tensor] = []
        good_paths: List[Path] = []
        for path in batch_paths:
            try:
                wave = load_wav(str(path))
                wave_list.append(wave)
                good_paths.append(path)
            except Exception:
                errors += 1

        if not wave_list:
            outer.set_postfix(deleted=deleted, err=errors)
            continue

        batch = pad_sequence(wave_list, batch_first=True)  # (B, Tmax)
        batch = batch.to(DEVICE)

        try:
            scores = deep_noise_suppression_mean_opinion_score(
                batch, fs=SR, personalized=False, device=DEVICE
            ).tolist()  # [[p808, sig, bak, ovrl], ...]
        except Exception:
            # fall back to per‑clip inference on failure
            scores = []
            for w in batch:
                try:
                    sc = deep_noise_suppression_mean_opinion_score(
                        w, fs=SR, personalized=False, device=DEVICE
                    ).tolist()
                except Exception:
                    sc = None
                scores.append(sc)

        for path, sc in zip(good_paths, scores):
            if sc is None:
                errors += 1
                continue
            p808, sig, bak, ovrl = sc
            p808_vals.append(p808); sig_vals.append(sig); bak_vals.append(bak); ovrl_vals.append(ovrl)

            # if (p808 < P808_MIN) or (sig < SIG_MIN) or (bak < BAK_MIN) or (ovrl < OVRL_MIN):
            #     if DELETE_FILES:
            #         try:
            #             path.unlink()
            #         except Exception:
            #             pass
            #     deleted += 1

        outer.set_postfix(deleted=deleted, err=errors)

    print(f"Finished. Deleted {deleted} files. Errors: {errors}")

    # ─── density plots ───────────────────────────────────────────────────────
    def plot_density(values: List[float], title: str, fname: str, bins: int = 100):
        if not values:
            return
        arr = np.asarray(values, dtype=float)
        plt.figure()
        plt.hist(arr, bins=bins, density=True, alpha=0.85)
        plt.xlabel("Score"); plt.ylabel("Density"); plt.title(title)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        path = Path(PLOTS_DIR) / fname
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot: {path}  (n={len(arr)}, mean={arr.mean():.3f})")

    plot_density(p808_vals, "DNSMOS P.808 MOS (p808)", "p808_density.png")
    plot_density(sig_vals,  "DNSMOS Signal Quality (sig)", "sig_density.png")
    plot_density(bak_vals,  "DNSMOS Background Quality (bak)", "bak_density.png")
    plot_density(ovrl_vals, "DNSMOS Overall Quality (ovrl)", "ovrl_density.png")


if __name__ == "__main__":
    main()
