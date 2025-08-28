import os
import torch
import bitsandbytes as bnb
from torch.utils.data import ConcatDataset, Dataset, RandomSampler, DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from loss_functions import mrstft_loss, complex_stft_loss
from utils import beep, grad_global_norm, clip_penalty,  load_datasets
from model import ConvTasNet
from tqdm import tqdm
import pickle

####### Hyperparameters #######
CONTINUE_FROM_CHECKPOINT=True
ITER=8
NUM_EPOCHS=91
BATCH_SIZE=6
LR = 5e-5
MAX_NORM = 10
SUBSET = 0.1  # fraction of the dataset to use per epoch
REVERB_DIRS = ["../../data/clean","../../data/reverbed/sports_centre_university_york"]
CLEAN_DIR= "../../data/clean"
###############################


def build_model(CONTINUE_FROM_CHECKPOINT):
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

    optimizer = get_optimizer(model)

    if CONTINUE_FROM_CHECKPOINT:
        # checkpoint_path = "../../checkpoints/v2/v2_model_epoch_"+str(ITER)+".pth"  # CHANGE to your checkpoint
        checkpoint_path = "../../checkpoints/v2/v2_model_epoch_8_-13.898161352912235.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    return model, optimizer

def get_optimizer(model):
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
        
        λ = 3.0         # tune this hyper-parameter
        threshold = 1.0

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            enhanced_wave = model(reverb_wave)
        excess = torch.relu(enhanced_wave.abs() - 1.0)
        loss_si_sdr = -si_sdr(enhanced_wave.float().squeeze(1), clean_wave.float().squeeze(1))
        loss_mr = mrstft_loss(enhanced_wave.squeeze(1), clean_wave.squeeze(1), 
                              fft_sizes=(512,1024,2048), hop_sizes=(128,256,512), win_lengths=(512,1024,2048))
        loss_com_mr = complex_stft_loss(enhanced_wave.squeeze(1), clean_wave.squeeze(1), 
                              fft_sizes=(512,1024,2048), hop_sizes=(128,256,512), win_lengths=(512,1024,2048))
        # loss_clip = clip_penalty(enhanced_wave, threshold)

        # loss = loss_si_sdr + λ * loss_clip
        loss = loss_si_sdr + λ * loss_mr + λ * loss_com_mr
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
    
    with open('../evaluate/loss_data/v2/v2_loss_epoch_0' + str(epoch+ITER+1) + '.pkl', 'wb') as file:
        pickle.dump(si_sdr_list, file)

    with open('../evaluate/loss_data/v2/v2_mr_loss_epoch_0' + str(epoch+ITER+1) + '.pkl', 'wb') as file:
        pickle.dump(mr_list, file)
        
    average_loss = total_loss/len(dloader)
    print(f"Epoch {epoch+ITER+1} finished with average batch loss={average_loss:.6f}")
    
    os.makedirs("../../checkpoints/v2", exist_ok=True)

    checkpoint_path = os.path.join("../../checkpoints/v2", f"v2_model_epoch_{epoch+ITER+1}_{average_loss}.pth")
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
