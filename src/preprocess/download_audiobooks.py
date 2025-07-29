import requests
import os
import re
import time

# Directory where audiobooks will be saved
DOWNLOAD_DIR = "librivox_audiobooks"

def safe_filename(text, max_length=100):
    """Create a filesystem safe filename from text."""
    # Remove any character that's not alphanumeric, space, dash or underscore.
    safe = re.sub(r'[^a-zA-Z0-9 \-_]', '', text)
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_length]

def download_file(url, destination, chunk_size=32768):
    """Download a file from a URL in streaming mode."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write in binary mode, downloading in chunks.
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url} -> {e}")
        return False

def download_audiobooks_catalog():
    base_url = "https://librivox.org/api/feed/audiobooks/"
    params = {
        "format": "json",
        "extended": "1",  # Get full set of details, including download URLs.
        "limit": 50,
        "offset": 0
    }
    
    # Create download folder if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    page = 0
    total_downloaded = 0

    while True:
        print(f"\nFetching page {page}, offset {params['offset']}...")
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch data from API: {e}")
            break

        data = response.json()
        # Assuming the list of audiobooks is stored in the "books" key.
        books = data.get("books", [])
        if not books:
            print("No more audiobooks found; finishing download.")
            break

        for book in books:
            book_id = book.get("id")
            title = book.get("title", "unknown_title")
            # Use a safe filename based on id and title
            safe_title = safe_filename(title)
            filename = f"{book_id}-{safe_title}.zip"
            file_path = os.path.join(DOWNLOAD_DIR, filename)

            # Look for the URL of the ZIP file
            zip_url = book.get("url_zip_file")
            if not zip_url:
                print(f"[{book_id}] '{title}' does not have a downloadable zip file. Skipping.")
                continue

            if os.path.exists(file_path):
                print(f"[{book_id}] '{title}' already downloaded. Skipping.")
                continue

            print(f"Downloading [{book_id}] '{title}' from {zip_url}...")
            success = download_file(zip_url, file_path)
            if success:
                print(f"Downloaded and saved to {file_path}")
                total_downloaded += 1
            else:
                print(f"Failed to download [{book_id}] '{title}'.")
            
            # Optional: pause briefly between downloads to be kind to the server.
            time.sleep(1)
        
        # If number of books in this page is less than the limit, we're done.
        if len(books) < params["limit"]:
            print("Reached the final page.")
            break

        # Move to the next page
        params["offset"] += params["limit"]
        page += 1

    print(f"\nFinished downloading. Total audiobooks downloaded: {total_downloaded}")

if __name__ == '__main__':
    download_audiobooks_catalog()
