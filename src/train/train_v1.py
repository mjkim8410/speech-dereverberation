import os
import torch
import torchaudio
import bitsandbytes as bnb
from torch.utils.data import ConcatDataset, Dataset, RandomSampler, DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from model import ConvTasNet
from tqdm import tqdm
import pickle

####### Hyperparameters #######
CONTINUE_FROM_CHECKPOINT=True
ITER=54
NUM_EPOCHS=61
BATCH_SIZE=20
LR = 1e-4
MAX_NORM = 10
SUBSET = 1  # fraction of the dataset to use per epoch
REVERB_DIRS = ["../../data/clean","../../data/reverb_sports-centre-university-york"]
CLEAN_DIR= "../../data/clean"
###############################

def beep():
    try:
        import winsound
        for i in range (2):
            winsound.Beep(1000, 300)          # call at the end of each epoch
        winsound.Beep(1500, 300)          # call after each epoch
        winsound.Beep(2000, 300)          # call after each epoch
    except RuntimeError:
        pass  # audio device unavailable (RDP, headless)

def grad_global_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)        # ‖g‖₂
            total += param_norm.item() ** 2
    return total ** 0.5

def clip_penalty(wave, threshold=1.0, p=3):
    excess = torch.relu(wave.abs() - threshold)*1e+2
    return (excess**p).mean()

def mrstft_loss(
    y_hat: torch.Tensor, y: torch.Tensor, sr=16000,
    fft_sizes=(512, 1024, 2048),
    hop_sizes=(128, 256, 512),
    win_lengths=(512, 1024, 2048),
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


class DereverbDataset(Dataset):
    def __init__(self, reverb_dir, clean_dir, sample_rate, chunk_size):
        self.reverb_dir = reverb_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.reverb_files = sorted(os.listdir(reverb_dir))
    
    def __len__(self):
        return len(self.reverb_files)
    
    def __getitem__(self, idx):
        rev_name = self.reverb_files[idx]
        rev_path = os.path.join(self.reverb_dir, rev_name)
        clean_path = os.path.join(self.clean_dir, rev_name)
        
        reverb_wave, sr1 = torchaudio.load(rev_path, backend="soundfile")
        clean_wave, sr2 = torchaudio.load(clean_path, backend="soundfile")
        
        total_samples = reverb_wave.shape[1]
        if total_samples > self.chunk_size:
            start = torch.randint(0, total_samples - self.chunk_size, (1,)).item()
            end = start + self.chunk_size
            reverb_wave = reverb_wave[:, start:end]
            clean_wave = clean_wave[:, start:end]
        
        return reverb_wave, clean_wave
    
def load_datasets(REVERB_DIRS, CLEAN_DIR):
    reverb_dirs = REVERB_DIRS
    clean_dir = CLEAN_DIR

    # dataset_A = DereverbDataset(
    #     reverb_dir=reverb_dirs[0],
    #     clean_dir=clean_dir,
    #     sample_rate=16000,
    #     chunk_size=10 * 16000
    # )

    dataset_B = DereverbDataset(
        reverb_dir=reverb_dirs[1],
        clean_dir=clean_dir,
        sample_rate=16000,
        chunk_size=10 * 16000
    )

    # dataset_C = DereverbDataset(
    #     reverb_dir=reverb_dirs[2],
    #     clean_dir=clean_dir,
    #     sample_rate=16000,
    #     chunk_size=10 * 16000
    # )

    # return ConcatDataset([dataset_A, dataset_B, dataset_C])
    # return ConcatDataset([dataset_B, dataset_C])
    return dataset_B

def build_model(CONTINUE_FROM_CHECKPOINT):
    model = ConvTasNet(
        num_sources=1,
        encoder_kernel_size=16,   # L
        encoder_stride=8,         # 50 % overlap
        encoder_filters=512,      # N
        tcn_hidden=128,           # B
        tcn_kernel_size=3,
        tcn_layers=8,             # X  (dilations 1…128)
        tcn_stacks=3,             # R  (24 residual blocks total)
        causal=False
    ).cuda()

    optimizer = get_optimizer(model)

    if CONTINUE_FROM_CHECKPOINT:
        checkpoint_path = "../../checkpoints/v1/v1_model_epoch_"+str(ITER)+".pth"  # CHANGE to your checkpoint
        checkpoint_path = "../../checkpoints/v1/v1_model_epoch_54_-8.673741748695463.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    return model, optimizer

def get_optimizer(model):
    ###############################################################################
    # build parameter groups for weight-decay vs. no-decay
    ###############################################################################
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1-D params (biases, norm scales) → **no** weight decay
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:                                   # conv / linear kernels
            decay.append(param)

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=LR,        # same hyper-params as AdamW
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    return optimizer

def train_one_epoch(model, optimizer, epoch):

    si_sdr = ScaleInvariantSignalDistortionRatio().cuda()

    subset_per_epoch = int(SUBSET * len(full_train))

    sampler = RandomSampler(
    full_train,
    replacement=False,
    num_samples=subset_per_epoch,
    )

    dloader = DataLoader(
    full_train,
    batch_size=BATCH_SIZE,
    sampler=sampler,      # <-- replaces shuffle=True
    num_workers=6,
    pin_memory=True,
    )

    # Create a tqdm iterator for the DataLoader
    loop = tqdm(dloader, total=len(dloader), desc=f"Epoch [{epoch+ITER}/{NUM_EPOCHS+ITER}]", unit="batch", position=0)
    current_loss = 0.0
    total_loss = 0.0
    total_excess = 0.0
    clip = 0
    iter = 0
    si_sdr_list = []
    mr_list = []
    
    for reverb_wave, clean_wave in loop:
        reverb_wave = reverb_wave.cuda(non_blocking=True)
        clean_wave  = clean_wave.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        λ = 2.0         # tune this hyper-parameter
        threshold = 1.0

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            enhanced_wave = model(reverb_wave)
        excess = torch.relu(enhanced_wave.abs() - 1.0)
        loss_si_sdr = -si_sdr(enhanced_wave.float().squeeze(1), clean_wave.float().squeeze(1))
        loss_mr = mrstft_loss(enhanced_wave.squeeze(1), clean_wave.squeeze(1), 
                              fft_sizes=(512,1024,2048), hop_sizes=(128,256,512), win_lengths=(512,1024,2048))
        # loss_clip = clip_penalty(enhanced_wave, threshold)

        # loss = loss_si_sdr + λ * loss_clip
        loss = loss_si_sdr + λ * loss_mr
        # loss = loss_si_sdr

        if excess.mean().item() > 0:
            clip += 1

        iter += 1
        current_loss = loss.item()
        total_loss += current_loss
        si_sdr_list.append(loss_si_sdr.item())
        mr_list.append(loss_mr.item())
        total_excess += excess.mean().item()
        average_loss = total_loss/iter
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        global_g = grad_global_norm(model) 
        optimizer.step()
        
        # Update progress bar
        loop.set_postfix(ls=current_loss, mr=loss_mr.item(), avg=average_loss, g=f"{global_g:.2e}", ex=f"{total_excess/len(dloader):.2e}")
    
    with open('../evaluate/loss_data/loss_epoch_0' + str(epoch+ITER+1) + '.pkl', 'wb') as file:
        pickle.dump(si_sdr_list, file)

    with open('../evaluate/loss_data/mr_loss_epoch_0' + str(epoch+ITER+1) + '.pkl', 'wb') as file:
        pickle.dump(mr_list, file)
        
    average_loss = total_loss/len(dloader)
    print(f"Epoch {epoch+ITER+1} finished with average batch loss={average_loss:.6f}")
    
    # -------------------------
    # SAVE CHECKPOINT
    # -------------------------

    # Create a folder for saving model checkpoints
    os.makedirs("../../checkpoints/v1", exist_ok=True)

    checkpoint_path = os.path.join("../../checkpoints/v1", f"v1_model_epoch_{epoch+ITER+1}_{average_loss}.pth")
    torch.save({
        "epoch": epoch+ITER+1,
        "train_avg_loss": average_loss, 
        "learning_rate": LR, 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    
if __name__ == "__main__":

    print("Loading datasets...")
    full_train = load_datasets(REVERB_DIRS, CLEAN_DIR)
    print(f"Total training samples: {len(full_train)}")
    print("Datasets loaded successfully.")

    model, optimizer = build_model(CONTINUE_FROM_CHECKPOINT)

    model.cuda()
    model.train()

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, epoch)
        # beep()
