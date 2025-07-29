import os
import zipfile

folder = 'librivox_audiobooks'
destination = 'librivox_audiobooks/audio'

for filename in os.listdir(folder):
    if filename.endswith('.zip'):
        zip_path = os.path.join(folder, filename)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print(filename, 'extracted.')
