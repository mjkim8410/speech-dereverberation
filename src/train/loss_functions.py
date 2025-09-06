import torch

def mrstft_loss(
    y_hat: torch.Tensor, y: torch.Tensor, sr=16000,
    fft_sizes   = (256, 512, 1024, 2048, 4096),
    hop_sizes   = (64, 128, 256, 512, 1024),
    win_lengths = (256, 512, 1024, 2048, 4096),
    alpha=0.5, beta=0.5, eps=1e-7, center=True
    ):
    """
    y_hat, y: [B, T] waveform tensors
    Returns scalar loss (mean over batch & resolutions).
    """
    assert y_hat.dim()==2 and y.dim()==2 and y_hat.shape==y.shape
    device = y_hat.device
    B = y_hat.size(0)

    total = 0.0
    n_res = len(fft_sizes)

    for n_fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(win_len, device=device)

        # Use fp32 for STFT math even under AMP
        Y  = torch.stft(y.float(),     n_fft=n_fft, hop_length=hop, win_length=win_len,
                        window=window, center=center, return_complex=True)
        Yh = torch.stft(y_hat.float(), n_fft=n_fft, hop_length=hop, win_length=win_len,
                        window=window, center=center, return_complex=True)

        mag  = Y.abs()
        magh = Yh.abs()

        # Spectral convergence
        sc_num = torch.linalg.norm(mag - magh, ord='fro', dim=(-2, -1))
        sc_den = torch.linalg.norm(mag,          ord='fro', dim=(-2, -1)) + eps
        L_sc   = (sc_num / sc_den).mean()

        # Log-mag L1
        L_log = (torch.log(mag + eps) - torch.log(magh + eps)).abs().mean()

        total += alpha * L_sc + beta * L_log

    return total / n_res

def complex_stft_loss(
    y_hat: torch.Tensor, y: torch.Tensor,
    fft_sizes   = (256, 512, 1024, 2048, 4096),
    hop_sizes   = (64, 128, 256, 512, 1024),
    win_lengths = (256, 512, 1024, 2048, 4096),
    center=True,
    eps=1e-8
    ):
    """
    Multi-resolution complex STFT loss (L2).
    Compares real+imag (phase-aware) STFTs between prediction and target.

    Args:
        y_hat, y: [B, T] waveforms
    Returns:
        scalar loss (mean over batch & resolutions)
    """
    assert y_hat.shape == y.shape
    device = y_hat.device
    total = 0.0
    n_res = len(fft_sizes)

    for n_fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths):
        win = torch.hann_window(win_len, device=device)

        Y  = torch.stft(y.float(),     n_fft, hop, win_len,
                        window=win, center=center, return_complex=True)
        Yh = torch.stft(y_hat.float(), n_fft, hop, win_len,
                        window=win, center=center, return_complex=True)

        diff = Y - Yh
        # L2 distance in complex domain
        loss = torch.sqrt(diff.real**2 + diff.imag**2 + eps).mean()

        total += loss

    return total / n_res
