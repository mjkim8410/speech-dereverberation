import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

input_folder = Path("loss_data")
window = 500

def load_loss_data(file_name):
    loss_list = []
    for p in sorted(input_folder.glob(file_name + "*.pkl")):
        with p.open("rb") as f:
            obj = pickle.load(f)
        # handle list or numpy array
        arr = np.asarray(obj, dtype=float).ravel()
        loss_list.extend(arr.tolist())
    return np.asarray(loss_list, dtype=float)

def smooth_loss(loss, window):
    # 2) Smooth efficiently (moving average)
    if len(loss) >= window:
        kernel = np.ones(window, dtype=float) / window
        smooth_loss = np.convolve(loss, kernel, mode="valid")  # length = N - window + 1
    else:
        smooth_loss = np.array([])
    return smooth_loss

def left_pad_to_match(arr: np.ndarray, target_len: int) -> np.ndarray:
    n = arr.size
    pad = np.full(target_len - n, np.nan, dtype=float)
    return np.concatenate([pad, arr])

def plot_series(x1, x2, title, fname):
    plt.figure()
    plt.plot(x1, alpha=0.85, label="SI-SDR")
    plt.plot(x2, alpha=0.85, label="MR-STFT")
    plt.xlabel("Batch"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(which='minor', linestyle=':', linewidth=0.4)
    plt.minorticks_on(); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved plot: {fname}  (n1={x1.size}, n2={x2.size}, mean1={np.nanmean(x1):.6f}, mean2={np.nanmean(x2):.6f})")

si_sdr_list = load_loss_data("v2_loss_epoch_")
mr_list = load_loss_data("v2_mr_loss_epoch_")

smooth_si_sdr = smooth_loss(si_sdr_list, window)
smooth_mr = smooth_loss(mr_list, window)

mr_list = left_pad_to_match(mr_list, si_sdr_list.size)
smooth_mr = left_pad_to_match(smooth_mr, smooth_si_sdr.size)

plot_series(si_sdr_list, mr_list, "Loss", "v2_loss.png")
plot_series(smooth_si_sdr, smooth_mr, f"Loss (moving avg, window={window})", "v2_smooth_loss.png")

smooth_mr = smooth_loss(smooth_mr, 2000)

plt.figure()
plt.plot(smooth_mr, alpha=0.85, label="MR-STFT")
plt.xlabel("Batch"); plt.ylabel("Loss"); plt.title("MR-STFT")
plt.grid(True, which='major', alpha=0.5)
plt.grid(which='minor', linestyle=':', linewidth=0.4)
plt.minorticks_on(); plt.legend(); plt.tight_layout()
plt.savefig("v2_smooth_mr_loss.png", dpi=150); plt.close()
print(f"Saved plot: {"v2_smooth_mr_loss.png"}  (n1={smooth_mr.size}, mean={np.nanmean(smooth_mr):.6f}")
