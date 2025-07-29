import os
import torch
import torchaudio
import bitsandbytes as bnb
from torch.utils.data import ConcatDataset, Dataset, RandomSampler, DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from model import ConvTasNet
from tqdm import tqdm

####### Hyperparameters #######
CONTINUE_FROM_CHECKPOINT=True
ITER=252
NUM_EPOCHS=10000
BATCH_SIZE=2
LR = 2e-07
SUBSET = 0.01  # fraction of the dataset to use per epoch
REVERB_DIRS = ["../../data/clean_chunks","../../data/reverb_chunks","../../data/reverb2_chunks"]
CLEAN_DIR= "../../data/clean_chunks"
LOSS = 99999
REDUCE_LR = False
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

    dataset_C = DereverbDataset(
        reverb_dir=reverb_dirs[2],
        clean_dir=clean_dir,
        sample_rate=16000,
        chunk_size=10 * 16000
    )

    # return ConcatDataset([dataset_A, dataset_B, dataset_C])
    return ConcatDataset([dataset_B, dataset_C])

def build_model(CONTINUE_FROM_CHECKPOINT):
    model = ConvTasNet(
        num_sources=1,
        encoder_kernel_size=32,
        encoder_stride=16,
        encoder_filters=2048,
        tcn_hidden=1024,
        tcn_kernel_size=7,
        tcn_layers=8,
        tcn_stacks=7,
        causal=False
    ).cuda()

    if CONTINUE_FROM_CHECKPOINT:
        checkpoint_path = "../../checkpoints_v3/v3_model_epoch_"+str(ITER)+".pth"  # CHANGE to your checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    return model

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

def train_one_epoch(model, optimizer, epoch, LOSS, REDUCE_LR):

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

    # print("Data loader created with subset size:", subset_per_epoch)

    # Create a tqdm iterator for the DataLoader
    loop = tqdm(dloader, total=len(dloader), desc=f"Epoch [{epoch+ITER}/{NUM_EPOCHS+ITER}]", unit="batch", position=0)
    current_loss = 0.0
    total_loss = 0.0
    total_excess = 0.0
    clip = 0
    iter = 0
    
    for reverb_wave, clean_wave in loop:
        reverb_wave = reverb_wave.cuda()
        clean_wave = clean_wave.cuda()

        optimizer.zero_grad()
        
        λ = 1.0e+5          # tune this hyper-parameter
        threshold = 1.0

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            enhanced_wave = model(reverb_wave)
            excess = torch.relu(enhanced_wave.abs() - 1.0)
            loss_main = -si_sdr(enhanced_wave, clean_wave)
            loss_clip = clip_penalty(enhanced_wave, threshold)

            # loss = loss_main + λ * loss_clip
            loss = loss_main

        if excess.mean().item() > 0:
            clip += 1

        iter += 1
        current_loss = loss.item()
        total_loss += current_loss
        total_excess += excess.mean().item()
        average_loss = total_loss/iter
        
        max_norm = 250.0 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        global_g = grad_global_norm(model) 
        optimizer.step()
        
        # Update progress bar
        loop.set_postfix(ls=current_loss, avg=average_loss, g=f"{global_g:.2e}", ex=f"{total_excess/len(dloader):.2e}")
    
    average_loss = total_loss/len(dloader)
    if average_loss < LOSS:
        LOSS = average_loss
    else:
        REDUCE_LR = True

    print(f"Epoch {epoch+ITER+1} finished with average batch loss={average_loss:.6f}")
    
    # -------------------------
    # SAVE CHECKPOINT
    # -------------------------

    # Create a folder for saving model checkpoints
    os.makedirs("../../checkpoints_v3", exist_ok=True)

    checkpoint_path = os.path.join("../../checkpoints_v3", f"v3_model_epoch_{epoch+ITER+1}.pth")
    torch.save({
        "epoch": epoch+ITER+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": current_loss
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    return LOSS, REDUCE_LR

    
if __name__ == "__main__":

    loss = 99999

    # print("Loading datasets...")
    full_train = load_datasets(REVERB_DIRS, CLEAN_DIR)
    # print(f"Total training samples: {len(full_train)}")
    # print("Datasets loaded successfully.")

    model = build_model(CONTINUE_FROM_CHECKPOINT)

    model.cuda()
    model.train()

    optimizer = get_optimizer(model)

    for epoch in range(NUM_EPOCHS):
        LOSS, REDUCE_LR = train_one_epoch(model, optimizer, epoch, LOSS, REDUCE_LR)
        if REDUCE_LR:
            LR = LR * 1
            REDUCE_LR = False

        LR = LR * 1
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        print(f"Learning rate updated to: {LR:.1e}")
        print(REDUCE_LR)
        print(f"Current loss: {LOSS:.6f}")
        # beep()
