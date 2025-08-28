import torch, torchaudio, os
from torch.utils.data import Dataset

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

    # dataset_C = DereverbDataset(
    #     reverb_dir=reverb_dirs[2],
    #     clean_dir=clean_dir,
    #     sample_rate=16000,
    #     chunk_size=10 * 16000
    # )

    # return ConcatDataset([dataset_A, dataset_B, dataset_C])
    # return ConcatDataset([dataset_B, dataset_C])
    return dataset_B