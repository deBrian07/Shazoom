import os
import csv
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
import concurrent.futures

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from utils.constants import DEV_MODE, MONGO_URI
from utils.utils import audio_file_to_samples, generate_fingerprints_multiresolution

def process_song(song, songs_col, fingerprints_col):
    if songs_col.find_one({"title": song["title"], "artist": song["artist"]}):
        return f"Skipping '{song['title']}' by {song['artist']}: Already exists."
    
    file_path = os.path.join("download", song["file"])
    try:
        with open(file_path, "rb") as f:
            samples, sr = audio_file_to_samples(f)
    except Exception as e:
        return f"Failed to load audio from {file_path}: {e}"
    
    # Use the new multi-resolution fingerprint function.
    fingerprints = generate_fingerprints_multiresolution(samples, sr)
    if not fingerprints:
        return f"Warning: No fingerprints generated for '{song['title']}'."
    
    song_doc = {"title": song["title"], "artist": song["artist"]}
    result = songs_col.insert_one(song_doc)
    song_id = result.inserted_id
    
    fp_docs = [{"song_id": song_id, "hash": h, "offset": offset} for (h, offset) in fingerprints]
    fingerprints_col.insert_many(fp_docs)
    
    return f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'."

def main():
    # Connect to MongoDB.
    client = MongoClient(MONGO_URI)

    if DEV_MODE:
        db = client["musicDB_dev"]
    else:
        db = client["musicDB"]
    
    songs_col = db["songs"]
    fingerprints_col = db["fingerprints"]
    # fingerprints_col.create_index("hash")
    
    csv_path = os.path.join("download", "processed.csv")
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
