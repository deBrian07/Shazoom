import os
import csv

# Directory where the WAV files are saved.
DOWNLOADS_DIR = "downloads"
# Output CSV file name.
OUTPUT_CSV = "processed.csv"

def extract_song_info(filename):
    """
    Given a filename (e.g., "song name - artist.wav"), 
    extract and return the song name and artist.
    If the expected format isn't met, the entire filename (sans extension)
    is returned as the song name and the artist will be 'Unknown'.
    """
    # Remove the file extension.
    base_name = os.path.splitext(filename)[0]
    # Attempt to split the base name into song name and artist.
    parts = base_name.split(" - ")
    if len(parts) >= 2:
        song_name = parts[0].strip()
        # In case there are additional dashes in the artist name
        artist = " - ".join(parts[1:]).strip()
    else:
        song_name = base_name.strip()
        artist = "Unknown"
    return song_name, artist

def get_processed_entries():
    """
    Scans the DOWNLOADS_DIR for WAV files and returns a list of tuples:
    (song name, artist, wav file location).
    """
    entries = []
    if not os.path.exists(DOWNLOADS_DIR):
        print(f"Directory '{DOWNLOADS_DIR}' does not exist.")
        return entries

    for fname in os.listdir(DOWNLOADS_DIR):
        if fname.lower().endswith(".wav"):
            song_name, artist = extract_song_info(fname)
            # Get the full (relative) file path.
            wav_path = os.path.join(DOWNLOADS_DIR, fname)
            entries.append((song_name, artist, wav_path))
    return entries

def write_processed_csv(entries):
    """
    Writes the processed entries (song name, artist, wav file location)
    to a CSV file defined by OUTPUT_CSV.
    """
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["song name", "artist", "wav file location"])
        writer.writerows(entries)
    print(f"CSV file '{OUTPUT_CSV}' written with {len(entries)} entries.")

def main():
    entries = get_processed_entries()
    if entries:
        write_processed_csv(entries)
    else:
        print("No WAV files were found to process.")

if __name__ == "__main__":
    main()
