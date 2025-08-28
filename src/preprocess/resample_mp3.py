import os
import torchaudio

input_dir = "../../data/test_audio_segmented"
output_dir = "../../data/test_audio"
sample_rate_target = 16000

os.makedirs(output_dir, exist_ok=True)

fnames = os.listdir(input_dir)
total = len(fnames)
counter = 0
for fname in fnames:
    print(f'Progress: {(counter*100 / total):.3f}% done')
    if fname.lower().endswith('.mp3'):
        input_path = os.path.join(input_dir, fname)
        wave, sr = torchaudio.load(input_path)
        
        wave = torchaudio.functional.resample(wave, sr, sample_rate_target)
        
        # Build output name
        base, ext = os.path.splitext(fname)
        out_fname = base + "_16k.wav"
        out_path = os.path.join(output_dir, out_fname)
        
        torchaudio.save(out_path, wave, sample_rate_target)
    counter += 1
