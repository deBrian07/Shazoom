#!/usr/bin/env python3
import csv
import sys
import os
import yt_dlp
from tqdm import tqdm

# Path to your cookies file for authenticated YouTube access
# Please export your YouTube cookies (Netscape format) from a browser on another machine
# and upload it to this server as 'cookies.txt'.
COOKIES_PATH = "cookies.txt"


def download_song(song_name, artist):
    """
    Searches YouTube for the given song and artist and downloads the audio in WAV format,
    using a pre-exported cookies file for authentication.
    """
    # Ensure cookies file exists
    if not os.path.exists(COOKIES_PATH):
        print(f"Error: Cookies file '{COOKIES_PATH}' not found.")
        print("Please export your YouTube cookies (Netscape format) and place them at this path.")
        sys.exit(1)

    query = f"{song_name} {artist}"
    search_query = f"ytsearch1:{query}"

    # Build target path, accounting for '/' in names creating subdirectories
    target_file = os.path.join("downloads", f"{song_name} - {artist}.wav")
    if os.path.exists(target_file):
        print(f"Skipping: '{song_name}' by '{artist}' already exists.")
        return

    print(f"Downloading: '{song_name}' by '{artist}'...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join("downloads", f"{song_name} - {artist}.%(ext)s"),
        'cookiefile': COOKIES_PATH,  # use uploaded cookies.txt
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])
        print(f"Completed: '{song_name}' by '{artist}'.")
    except Exception as e:
        print(f"Error downloading '{song_name}' by '{artist}': {e}")


def process_csv(csv_filepath):
    """
    Reads the CSV of song and artist columns and downloads each entry.
    """
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
