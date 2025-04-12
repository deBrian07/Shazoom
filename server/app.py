import os
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from collections import defaultdict, Counter
from database.utils import (
    audio_file_to_samples,
    generate_fingerprints_multiresolution,
    accumulate_votes_for_hash,
    merge_votes
)
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

app = Flask(__name__)
allowed_origins = ["http://localhost:3000", "https://debrian07.github.io"]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

# MongoDB connection string
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

DEV_MODE = False  # Change to True for testing if needed.
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]
fingerprints_col.create_index("hash")

def find_fingerprint_batch(hash_batch):
    """Query fingerprints using a batch of hashes and return the list of docs."""
    cursor = fingerprints_col.find(
        {"hash": {"$in": hash_batch}},
        {"hash": 1, "song_id": 1, "offset": 1}
    ).batch_size(1000)
    return list(cursor)

@app.route('/identify', methods=['POST'])
def identify_song():
    start_time = time.time()
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files["audio"]
    try:
        samples, sample_rate = audio_file_to_samples(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Generate fingerprints using the multi-resolution method.
    query_fps = generate_fingerprints_multiresolution(samples, sample_rate)
    if not query_fps:
        return jsonify({"error": "No fingerprints generated from audio."}), 500

    # Build mapping: fingerprint hash --> list of candidate offsets.
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)
    hash_list = list(query_hashes.keys())
    
    # --- Improved Query: Split hash_list into batches and query concurrently ---
    batch_size = 16  # you may tune this value based on performance
    hash_batches = [hash_list[i:i+batch_size] for i in range(0, len(hash_list), batch_size)]
    num_batches = len(hash_batches)
    
    start_find = time.time()
    # Use ThreadPoolExecutor to run parallel queries.
    with ThreadPoolExecutor(max_workers=max(128, num_batches)) as executor:
        futures = [executor.submit(find_fingerprint_batch, batch) for batch in hash_batches]
        results_batches = [future.result() for future in futures]
    db_docs = list(chain.from_iterable(results_batches))
    find_query_time = time.time() - start_find

    # Group the DB fingerprints by hash.
    db_group = defaultdict(list)
    for doc in db_docs:
        h = doc["hash"]
        db_group[h].append((doc["song_id"], doc["offset"]))

    bin_width = 0.2  # seconds; adjustable for offset binning
    global_votes = defaultdict(int)

    start_match = time.time()
    # For each fingerprint hash that appears in the query
    for h, query_offsets in query_hashes.items():
        if h not in db_group:
            continue
        query_offsets_array = np.array(query_offsets, dtype=np.float64)
        for song_id, db_offset in db_group[h]:
            votes_for_hash = accumulate_votes_for_hash(query_offsets_array, db_offset, bin_width)
            merge_votes(global_votes, votes_for_hash, song_id)
    match_time = time.time() - start_match

    if not global_votes:
        overall_time = time.time() - start_time
        return jsonify({
            "result": "No match found.",
            "find_query_time": find_query_time,
            "match_time": match_time,
            "overall_time": overall_time
        }), 200

    vote_counts = Counter(global_votes)
    best_match, best_votes = vote_counts.most_common(1)[0]
    best_song_id, best_delta = best_match

    MIN_VOTES = 30
    if best_votes < MIN_VOTES:
        overall_time = time.time() - start_time
        return jsonify({
            "result": "No match found.",
            "find_query_time": find_query_time,
            "match_time": match_time,
            "overall_time": overall_time
        }), 200

    song = songs_col.find_one({"_id": best_song_id})
    if not song:
        return jsonify({"result": "No match found."}), 200

    total_query = len(query_fps)
    vote_ratio = best_votes / total_query if total_query > 0 else 0

    overall_time = time.time() - start_time
    return jsonify({
        "song": song.get("title"),
        "artist": song.get("artist"),
        "offset": best_delta,
        "raw_votes": best_votes,
        "vote_ratio": vote_ratio,
        "find_query_time": find_query_time,
        "match_time": match_time,
        "overall_time": overall_time
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
