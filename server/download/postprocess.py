import os
import csv

# Directory where WAV files are stored.
DOWNLOADS_DIR = "downloads"
# Output CSV file.
OUTPUT_CSV = "processed.csv"

def extract_song_info_from_relpath(rel_path):
    """
    Given a relative path (using "/" as separator) representing the song's full name—for example:
      "Thunderstruck - AC/DC.wav"—
    this function removes the '.wav' extension and splits on the first occurrence of " - "
    to return the song name and artist.
    
    If the pattern doesn't match, it returns the whole string as song name and marks the artist as "Unknown".
    """
    # Remove the .wav extension (case-insensitive)
    if rel_path.lower().endswith(".wav"):
        rel_path = rel_path[:-4]
    # Attempt to split by the separator " - " (only on the first occurrence)
    if " - " in rel_path:
        song_name, artist = rel_path.split(" - ", 1)
        return song_name.strip(), artist.strip()
    else:
        return rel_path.strip(), "Unknown"

def get_processed_entries():
    """
    Recursively scan the DOWNLOADS_DIR for WAV files.
    For each found file, reconstruct the intended filename by converting the relative path to use "/"
    as separator, then parse out the song name and artist.
    
    Returns:
      A list of tuples: (song name, artist, full file path)
    """
    entries = []
    if not os.path.exists(DOWNLOADS_DIR):
        print(f"Directory '{DOWNLOADS_DIR}' does not exist.")
        return entries

    # Walk the downloads directory recursively.
    for root, dirs, files in os.walk(DOWNLOADS_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                # Get the full file path.
                file_path = os.path.join(root, file)
                # Get the relative path with respect to DOWNLOADS_DIR.
                rel_path = os.path.relpath(file_path, DOWNLOADS_DIR)
                # Convert any OS-specific separators to "/" to reconstruct the intended name.
                rel_path_normalized = rel_path.replace(os.sep, "/")
                # Extract song name and artist.
                song_name, artist = extract_song_info_from_relpath(rel_path_normalized)
                entries.append((song_name, artist, file_path))
    return entries

def write_processed_csv(entries, output_csv=OUTPUT_CSV):
    """
    Writes the list of entries (song name, artist, wav file location) to a CSV file.
    """
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["song name", "artist", "wav file location"])
        writer.writerows(entries)
    print(f"CSV file '{output_csv}' written with {len(entries)} entries.")

def main():
    entries = get_processed_entries()
    if entries:
        write_processed_csv(entries)
    else:
        print("No WAV files found to process.")

if __name__ == "__main__":
    main()
