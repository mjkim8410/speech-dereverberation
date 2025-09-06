import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

input_folder = Path("loss_data/v2")
out_dir = Path("eval_plots")
out_dir.mkdir(parents=True, exist_ok=True)

window = 200  # moving-average window

def load_loss_data(prefix: str) -> np.ndarray:
    """Load and concatenate all pickles matching <prefix>*.pkl into one 1D float array."""
    loss_list = []
    for p in sorted(input_folder.glob(prefix + "*.pkl")):
        with p.open("rb") as f:
            obj = pickle.load(f)
        arr = np.asarray(obj, dtype=float).ravel()
        loss_list.extend(arr.tolist())
    return np.asarray(loss_list, dtype=float)

def smooth_loss(arr: np.ndarray, win: int) -> np.ndarray:
    """Centered-ish moving average via 'valid' conv, then left-pad with NaNs to match length."""
    if arr.size < win:
        return np.full(arr.size, np.nan, dtype=float)
    kernel = np.ones(win, dtype=float) / win
    sm = np.convolve(arr, kernel, mode="valid")  # length = N - win + 1
    pad = np.full(arr.size - sm.size, np.nan, dtype=float)
    return np.concatenate([pad, sm])

def plot_raw_and_smooth(y_raw: np.ndarray, y_smooth: np.ndarray, title: str, fname: Path, ylabel="Loss"):
    plt.figure()
    plt.plot(y_raw,   alpha=0.45, label="raw")
    plt.plot(y_smooth, alpha=0.95, label=f"moving avg (window={window})")
    plt.xlabel("Batch"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, which="major", alpha=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.4)
    plt.minorticks_on(); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved plot: {fname}  (n={y_raw.size}, mean_raw={np.nanmean(y_raw):.6f}, mean_smooth={np.nanmean(y_smooth):.6f})")

# ---- Load all four metrics ----
si_sdr_list  = load_loss_data("v2_loss_epoch_")
mr_list      = load_loss_data("v2_mr_loss_epoch_")
com_mr_list  = load_loss_data("v2_com_mr_loss_epoch_")
mse_list     = load_loss_data("v2_mse_loss_epoch_")

# ---- Smooth each ----
smooth_si    = smooth_loss(si_sdr_list, window)
smooth_mr    = smooth_loss(mr_list,     window)
smooth_cmr   = smooth_loss(com_mr_list, window)
smooth_mse   = smooth_loss(mse_list,    window)

# ---- Plot one figure per metric (4 total) ----
plot_raw_and_smooth(si_sdr_list, smooth_si,  "SI-SDR loss (raw + moving average)", out_dir / "v2_si_sdr.png", ylabel="Loss")
plot_raw_and_smooth(mr_list,     smooth_mr,  "MR-STFT loss (raw + moving average)", out_dir / "v2_mr.png",     ylabel="Loss")
plot_raw_and_smooth(com_mr_list, smooth_cmr, "Complex MR-STFT loss (raw + moving average)", out_dir / "v2_complex_mr.png", ylabel="Loss")
plot_raw_and_smooth(mse_list,    smooth_mse, "MSE loss (raw + moving average)", out_dir / "v2_mse.png",       ylabel="Loss")
