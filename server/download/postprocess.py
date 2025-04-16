import os
import csv
from tqdm import tqdm

# Directory where the WAV files are stored.
DOWNLOADS_DIR = "downloads"
# Output CSV file.
OUTPUT_CSV = "processed.csv"

def extract_song_info_from_relpath(rel_path):
    """
    Given a relative path (using "/" as separator) that represents the intended song name—
    for example, "Thunderstruck - AC/DC.wav"—this function removes the '.wav' extension,
    then splits only on the first occurrence of " - " to return (song name, artist).
    If the expected pattern isn't found, the artist is returned as 'Unknown'.
    """
    if rel_path.lower().endswith(".wav"):
        rel_path = rel_path[:-4]
    if " - " in rel_path:
        song_name, artist = rel_path.split(" - ", 1)
        return song_name.strip(), artist.strip()
    else:
        return rel_path.strip(), "Unknown"

def get_processed_entries():
    """
    Returns:
      A list of tuples: (song name, artist, full file path)
    """
    entries = {} 
    wav_files = []
    
    for root, dirs, files in os.walk(DOWNLOADS_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)
    
    for file_path in tqdm(wav_files, desc="Processing WAV files", unit="file"):
        rel_path = os.path.relpath(file_path, DOWNLOADS_DIR)
        rel_path_normalized = rel_path.replace(os.sep, "/")
        song_name, artist = extract_song_info_from_relpath(rel_path_normalized)
        key = (song_name, artist)
        if key not in entries:
            entries[key] = file_path
        else:
            print(f"Duplicate found: {song_name} - {artist} already processed, skipping '{file_path}'.")
    
    return [(song, artist, path) for (song, artist), path in entries.items()]

def write_processed_csv(entries, output_csv=OUTPUT_CSV):
    """
    Writes the processed entries (song name, artist, wav file location) into a CSV file.
    """
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["song name", "artist", "wav file location"])
        writer.writerows(entries)
    print(f"CSV file '{output_csv}' written with {len(entries)} unique entries.")

def main():
    entries = get_processed_entries()
    if entries:
        write_processed_csv(entries)
    else:
        print("No WAV files found to process.")

if __name__ == "__main__":
    main()
