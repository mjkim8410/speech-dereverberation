# dnsmos_batch_with_progress.py
# Requirements: pip install torchmetrics[audio] librosa onnxruntime tqdm matplotlib numpy
# (On NVIDIA GPU you may prefer: pip install onnxruntime-gpu)

import os
import glob
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

# -------- settings --------
FILES_GLOB = "../../data/clean_chunks/*.mp3"   # change to your folder
# FILES_GLOB = "../../data/audio_segmented/*.mp3"   
PLOTS_DIR = "./dnsmos_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
# --------------------------

p808_list, sig_list, bak_list, ovrl_list = [], [], [], []

files = sorted(glob.glob(FILES_GLOB))
if not files:
    print(f"No files matched: {FILES_GLOB}")
    raise SystemExit(0)

pbar = tqdm(files, desc="Scoring DNSMOS", unit="file")
err = 0

for f in pbar:
    try:
        # Load MP3 → mono 16 kHz float32 in [-1, 1]
        y, _ = librosa.load(f, sr=16000, mono=True)
        y_t = torch.tensor(y, dtype=torch.float32)

        # DNSMOS returns [p808, sig, bak, ovrl]
        p808, sig, bak, ovrl = deep_noise_suppression_mean_opinion_score(
            y_t, fs=16000, personalized=False
        ).tolist()

        p808_list.append(p808)
        sig_list.append(sig)
        bak_list.append(bak)
        ovrl_list.append(ovrl)

        # Show running averages in the progress bar
        pbar.set_postfix(
            err=err
        )
    except Exception as e:
        err += 1
        pbar.set_postfix(err=err, msg=f"error:{type(e).__name__}")

print(f"\nDone., ERR={err}")

def plot_density(values, title, filename, bins=100):
    if len(values) == 0:
        print(f"Skip plotting {title}: no data")
        return
    arr = np.asarray(values, dtype=float)
    plt.figure()
    plt.hist(arr, bins=bins, density=True, alpha=0.85)  # density histogram
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}  (n={len(arr)}, mean={arr.mean():.3f}, std={arr.std():.3f})")

# One chart per metric (no subplots)
plot_density(p808_list, "DNSMOS P.808 MOS (p808)", "p808_density.png")
plot_density(sig_list,  "DNSMOS Signal Quality (sig)", "sig_density.png")
plot_density(bak_list,  "DNSMOS Background Quality (bak)", "bak_density.png")
plot_density(ovrl_list, "DNSMOS Overall Quality (ovrl)", "ovrl_density.png")
