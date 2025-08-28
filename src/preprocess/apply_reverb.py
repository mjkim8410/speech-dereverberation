import os
import subprocess

def apply_afir_batch(
    input_folder,
    ir_file,
    output_folder,
    wet,
    dry,
    irgain,
):
    """
    Applies FFmpeg's afir filter to every audio file in 'input_folder' 
    using the provided impulse response 'ir_file'.
    
    :param input_folder:   Path to folder containing dry input audio files.
    :param ir_file:        Path to the impulse response (IR) audio file.
    :param output_folder:  Folder to save processed files. If None, uses input_folder.
    :param wet:            Wet level for afir.
    :param dry:            Dry level for afir.
    :param dry:            If 1, normalizes IR to unit energy.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the afir filter string
    filter_str = f"afir=wet={wet}:dry={dry}:irgain={irgain}"
    
    files = os.listdir(input_folder)
    total = len(files)
    counter = 0

    # Loop through all files in input_folder
    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        
        # Build output name (e.g., "example_afir.wav")
        base, ext = os.path.splitext(file_name)
        output_name = f"{base}_afir{ext}"
        output_path = os.path.join(output_folder, output_name)

        # Construct FFmpeg command
        cmd = [
            "ffmpeg", 
            "-hide_banner",          # drop the start-up banner
            "-loglevel", "error",    # only show errors (or use: warning / info / quiet / panic)
            "-nostats",
            "-y",
            "-i", input_path,
            "-i", ir_file,
            "-filter_complex", filter_str,
            "-c:a", "pcm_s16le", 
            output_path
        ]
        
        # Run the command
        subprocess.run(cmd, check=True)
        counter += 1
        print(str(counter) + "/" + str(total) + " done")
    
    print("All files have been processed.")

if __name__ == "__main__":
    dry_audio_folder = "../../data/clean"
    ir_wave = "../../data/IR/trimmed/1st_baptist_nashville_balcony_trimmed.wav"
    out_folder = "../../data/reverbed/1st_baptist_nashville_balcony_10_10_0.1"
    
    apply_afir_batch(
        input_folder=dry_audio_folder,
        ir_file=ir_wave,
        output_folder=out_folder,
        wet=10, 
        dry=10,
        irgain=0.1
    )
