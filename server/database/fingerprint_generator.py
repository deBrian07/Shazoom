import os
import csv
import numpy as np
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal
from scipy.ndimage import maximum_filter
from tqdm import tqdm
import concurrent.futures
from utils import audio_file_to_samples, generate_fingerprints

def process_song(song, songs_col, fingerprints_col):
    """
    Processes a single song:
      - Checks for duplicates.
      - Loads the audio file.
      - Generates fingerprints using the enhanced function.
      - Inserts song metadata and fingerprint documents into MongoDB.
    
    Returns a message.
    """
    if songs_col.find_one({"title": song["title"], "artist": song["artist"]}):
        return f"Skipping '{song['title']}' by {song['artist']}: Already exists."
    
    file_path = song["file"]
    try:
        with open(file_path, "rb") as f:
            samples, sr = audio_file_to_samples(f)
    except Exception as e:
        return f"Failed to load audio from {file_path}: {e}"
    
    fingerprints = generate_fingerprints(samples, sr)
    if not fingerprints:
        return f"Warning: No fingerprints generated for '{song['title']}'."
    
    song_doc = {"title": song["title"], "artist": song["artist"]}
    result = songs_col.insert_one(song_doc)
    song_id = result.inserted_id
    
    fp_docs = [{"song_id": song_id, "hash": hash_val, "offset": offset}
               for (hash_val, offset) in fingerprints]
    fingerprints_col.insert_many(fp_docs)
    
    return f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'."

def main():
    # Connect to MongoDB.
    MONGO_URI = "mongodb://localhost:27017"
    client = MongoClient(MONGO_URI)

    DEV_MODE = False
    if DEV_MODE:
        db = client["musicDB_dev"]
    else:
        db = client["musicDB"]
    
    songs_col = db["songs"]
    fingerprints_col = db["fingerprints"]
    fingerprints_col.create_index("hash")
    
    # Read CSV file with headers: "song name", "artist", "wav file location"
    csv_path = os.path.join("..", "download", "processed.csv")
    songs_to_add = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                songs_to_add.append({
                    "title": row["song name"].strip(),
                    "artist": row["artist"].strip(),
                    "file": row["wav file location"].strip()
                })
    except Exception as e:
        print(f"Failed to read CSV file at '{csv_path}': {e}")
        return
    
    results = []
    # Process songs concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_song, song, songs_col, fingerprints_col)
                   for song in songs_to_add]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Songs"):
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()
