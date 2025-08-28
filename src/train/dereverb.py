import os, torch, torchaudio
from model import ConvTasNet

# ------------------------------------------------------------------
# 1.  Save helper  (WAV, 32-bit float, no pydub/ffmpeg needed)
# ------------------------------------------------------------------
def save_tensor_as_wav(
    wave: torch.Tensor,
    sr: int = 16_000,
    out_path: str = "../../out/enhanced.wav",
    bits: int = 32,                     # 32-bit float keeps full dynamic range
):
    """
    Accepts (B,C,T), (C,T), (B,T) or (T,) tensors in [-1,1] and writes WAV.
    * bits = 32 → IEEE float
    * bits = 16 → PCM16
    """
    wave = wave.detach().cpu()

    # strip batch dim if present
    if wave.dim() == 3:                # (B,C,T)
        wave = wave.squeeze(0)         # -> (C,T)
    elif wave.dim() == 2 and wave.shape[0] == 1:  # (B,T) with B=1
        wave = wave.squeeze(0)         # -> (T,)

    if wave.dim() == 1:                # (T,) → (1,T)
        wave = wave.unsqueeze(0)

    # convert to float32 (required for 32-bit float WAV)
    wave = wave.to(torch.float32)

    torchaudio.save(out_path, wave, sr, bits_per_sample=bits)
    print(f"✔ saved {out_path} ({wave.shape[1]/sr:.2f}s, mono, {bits}-bit)")


# ------------------------------------------------------------------
# 2.  Load helper (unchanged – still uses torchaudio)
# ------------------------------------------------------------------
def load_audio_as_tensor(path: str, sr_target: int = 16_000) -> torch.Tensor:
    for backend in ("ffmpeg", "soundfile"):
        if backend in torchaudio.list_audio_backends():
            wav, sr = torchaudio.load(path, backend=backend)
            break
    else:
        raise RuntimeError("Install FFmpeg or soundfile so torchaudio can read audio.")
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav if wav.dim() == 2 else wav.unsqueeze(0)   # (1, T)


# ------------------------------------------------------------------
# 3.  Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    ckpt = torch.load("../../checkpoints/v2/v2_model_epoch_8_-13.898161352912235.pth", map_location="cpu")
    model = ConvTasNet(
        num_sources=1,
        encoder_kernel_size=32,
        encoder_stride=16,
        encoder_filters=1024,
        tcn_hidden=512,
        tcn_kernel_size=7,
        tcn_layers=7,
        tcn_stacks=7,
        causal=False
    ).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded checkpoint epoch", ckpt.get("epoch", "?"))
    model.cuda().eval()

    wav_in = load_audio_as_tensor("../../data/test_audio/01-Marxism lecture by Prof Raymond Geuss 5_8-consolidated_0017_16k.wav").cuda()
    print(wav_in)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        wav_out = model(wav_in)
    if wav_out.dim() == 3:
        wav_out = wav_out.squeeze(1)

    # ----------------------------------------------------------------
    # 4.  Save original, enhanced, and residual as 32-bit WAV
    # ----------------------------------------------------------------
    os.makedirs("../../out", exist_ok=True)

    save_tensor_as_wav(wav_in, 16_000, "../../out/original_input.wav", bits=32)
    save_tensor_as_wav(wav_out, 16_000, "../../out/enhanced_output.wav", bits=32)

    residual = wav_in - wav_out
    save_tensor_as_wav(residual, 16_000, "../../out/residual_output.wav", bits=32)





