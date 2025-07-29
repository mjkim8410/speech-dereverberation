from pathlib import Path
import torchaudio, torch, multiprocessing as mp

SRC  = Path("../../data/audio")
DST  = Path("../../data/audio_segmented")
DST.mkdir(exist_ok=True)

SEGMENT_SEC   = 10
CHUNK_SUFFIX  = "_{idx:04d}.mp3"   # change ".wav" → ".mp3" if you prefer

def process_one(path: Path):
    # --- fast header-only duration check ---
    info = torchaudio.info(path)
    duration = info.num_frames / info.sample_rate
    if duration < SEGMENT_SEC:
        print(f"[skip] {path.name}  ({duration} sec)  - deleting")
        path.unlink(missing_ok=True)                 # remove original file
        return

    # --- load + slice ---
    wave, sr = torchaudio.load(path)                 # float32 [-1,1]
    step      = sr * SEGMENT_SEC
    num_segs  = wave.size(1) // step

    base = path.stem
    for i in range(num_segs):
        chunk = wave[:, i*step : (i+1)*step]
        out   = DST / (base + CHUNK_SUFFIX.format(idx=i))
        torchaudio.save(out.as_posix(), chunk, sr)

    # tail shorter than 10 s is discarded automatically
    print(f"[ok]   {path.name} → {num_segs} chunks")

if __name__ == "__main__":
    files = list(SRC.glob("*.mp3"))
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_one, files)
