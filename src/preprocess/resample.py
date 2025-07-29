from pydub import AudioSegment
import os

input_dir = "audio"
output_dir = "resampled_audio"

os.makedirs(output_dir, exist_ok=True)

def resample_to_mp3(input_dir, output_dir, sample_rate, channels, bitrate):
    fnames = os.listdir(input_dir)
    total = len(fnames)
    counter = 0
    for fname in fnames:
        print(f'Progress: {(counter*100 / total):.3f}% done')

        # Load audio
        audio = AudioSegment.from_file(fname)
        
        # Convert channels if needed
        if audio.channels != channels:
            audio = audio.set_channels(channels)
        
        audio = audio.set_frame_rate(sample_rate)
        
        # Export to MP3
        # bitrate param helps define the output file size & quality
        audio.export(output_dir, format="mp3", bitrate=bitrate)
        counter += 1

# Usage
resample_to_mp3(input_dir, output_dir, 16000, 1, "64k")
print("Done!")
