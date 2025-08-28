import os

def remove_afir_suffix(folder_path):
    """
    Renames all files in 'folder_path' by removing '_afir' from their names.
    E.g. 'speech_afir.mp3' -> 'speech.mp3'
    """
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        if not os.path.isfile(old_path):
            continue  # skip directories or anything not a file
        
        # Split extension from base name
        base, ext = os.path.splitext(filename)
        
        # If filename includes "_afir" in the base
        # e.g. base="speech_afir", ext=".mp3"
        if base.endswith("_afir"):
            new_base = base[:-5]
            # Remove the substring "_afir"
            new_filename = new_base + ext
            
            new_path = os.path.join(folder_path, new_filename)
            
            # Check if a file with the new name already exists
            if os.path.exists(new_path):
                print(f"WARNING: {new_filename} already exists. Skipping rename to avoid overwrite.")
            else:
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    reverb_folder = '../../data/reverb_sports-centre-university-york'
    remove_afir_suffix(reverb_folder)
