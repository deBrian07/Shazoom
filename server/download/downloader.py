import csv
import sys
import os
import yt_dlp
from tqdm import tqdm

def download_song(song_name, artist):
    """
    Given a song name and artist, search YouTube and download the audio in WAV format.
    The function checks first if the expected target file exists (even if stored in nested directories)
    and skips the download if it already exists.
    """
    # Combine the song and artist to form the search query.
    query = f"{song_name} {artist}"
    search_query = f"ytsearch1:{query}"
    
    # Compute the expected output filename.
    # Note: if song name or artist contain a "/", this will create subdirectories accordingly.
    target_file = os.path.join("downloads", f"{song_name} - {artist}.wav")
    if os.path.exists(target_file):
        print(f"Skipping download: '{target_file}' already exists.")
        return
    
    # Setup yt_dlp options for downloading and converting audio to WAV.
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # Output template creates the same structure as target_file.
        'outtmpl': os.path.join("downloads", f"{song_name} - {artist}.%(ext)s"),
        'quiet': False,
        'no_warnings': True,
    }
    
    print(f"Searching and downloading: '{song_name}' by {artist}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])
        print(f"Download completed: '{song_name} - {artist}.wav'\n")
    except Exception as e:
        print(f"Error downloading '{song_name}' by {artist}: {e}\n")

def process_csv(csv_filepath):
    """
    Processes the CSV file that contains rows of song name and artist.
    Expects a header row with at least two columns:
      - 'song name'
      - 'artist'
    Displays a progress bar with tqdm.
    """
    # Ensure that the downloads directory exists.
    os.makedirs("downloads", exist_ok=True)
    
    with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        for row in tqdm(reader, desc="Downloading songs", unit="song"):
            try:
                song_name = row['song name'].strip()
                artist = row['artist'].strip()
            except KeyError:
                print("CSV file must contain 'song name' and 'artist' columns.")
                sys.exit(1)
            download_song(song_name, artist)

def main():
    csv_filepath = "songs.csv"
    if not os.path.exists(csv_filepath):
        print(f"Error: File '{csv_filepath}' does not exist.")
        sys.exit(1)
    
    process_csv(csv_filepath)

if __name__ == '__main__':
    main()
