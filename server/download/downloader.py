import csv
import sys
import os
import yt_dlp

def download_song(song_name, artist):
    """
    Given a song name and artist, search YouTube and download the audio in WAV format.
    """
    # Combine the song and artist to form the search query.
    query = f"{song_name} {artist}"
    search_query = f"ytsearch1:{query}"
    
    # Set up yt_dlp options to extract the best audio,
    # then convert the result to WAV using ffmpeg.
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # Output template: song name - artist.wav
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
    Process the CSV file that contains rows of song name and artist.
    
    Expects a header row with at least two columns:
      - 'song name' (or similar)
      - 'artist'
    """
    # Ensure the downloads output directory exists
    os.makedirs("downloads", exist_ok=True)
    
    with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
        # Adjust fieldnames if your CSV does not include a header row:
        # For CSV with header, csv.DictReader will use them.
        reader = csv.DictReader(csvfile)
        # If your CSV columns are labeled differently, update these keys.
        for row in reader:
            # Assuming the CSV header has the columns "song name" and "artist"
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
