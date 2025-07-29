import os
import subprocess

def apply_afir_batch(
    input_folder,
    ir_file,
    output_folder,
    wet,
    dry,
    audio_extensions=(".wav", ".mp3", ".flac", ".ogg")
):
    """
    Applies FFmpeg's afir filter to every audio file in 'input_folder' 
    using the provided impulse response 'ir_file'.
    
    :param input_folder:   Path to folder containing dry input audio files.
    :param ir_file:        Path to the impulse response (IR) audio file.
    :param output_folder:  Folder to save processed files. If None, uses input_folder.
    :param wet:            Wet level for afir.
    :param dry:            Dry level for afir.
    :param audio_extensions: File extensions to look for in input_folder.
    """
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the afir filter string
    filter_str = f"afir=wet={wet}:dry={dry},volume=7.0"
    
    # Loop through all files in input_folder
    for file_name in os.listdir(input_folder):
        # Check if file has an audio extension
        if file_name.lower().endswith(audio_extensions):
            input_path = os.path.join(input_folder, file_name)
            
            # Build output name (e.g., "example_afir.wav")
            base, ext = os.path.splitext(file_name)
            output_name = f"{base}_afir{ext}"
            output_path = os.path.join(output_folder, output_name)

            # Construct FFmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-i", ir_file,
                "-filter_complex", filter_str,
                "-c:a", "libmp3lame", 
                "-b:a", "64k",
                output_path
            ]
            
            print(f"Processing: {file_name} with IR: {os.path.basename(ir_file)}")
            
            # Run the command
            subprocess.run(cmd, check=True)
    
    print("All files have been processed.")

if __name__ == "__main__":
    dry_audio_folder = "librivox_audiobooks/audio"
    ir_wave = "librivox_audiobooks/IR/S2R1_M30.wav"
    out_folder = "librivox_audiobooks/reverb3"
    
    apply_afir_batch(
        input_folder=dry_audio_folder,
        ir_file=ir_wave,
        output_folder=out_folder,
        wet=2.0,
        dry=7.0
    )
