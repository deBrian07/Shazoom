import os
import csv
import numpy as np
from quart import Quart, request, jsonify
from quart_cors import cors
from motor.motor_asyncio import AsyncIOMotorClient
from collections import defaultdict, Counter
from database.utils import (
    audio_file_to_samples,
    generate_fingerprints_multiresolution,
    accumulate_votes_for_hash,
    merge_votes
)
import time
import asyncio
from itertools import chain

app = Quart(__name__)
allowed_origins = ["http://localhost:3000", "https://debrian07.github.io"]
app = cors(app, allow_origin=allowed_origins)

# MongoDB connection using Motor.
MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
DEV_MODE = False
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]

# Ensure the hash index exists.
@app.before_serving
async def ensure_index():
    await fingerprints_col.create_index("hash")

async def find_fingerprint_batch(batch):
    cursor = fingerprints_col.find(
        {"hash": {"$in": batch}},
        {"hash": 1, "song_id": 1, "offset": 1}
    ).batch_size(1000)
    return await cursor.to_list(length=None)

@app.route('/identify', methods=['POST'])
async def identify_song():
    start_time = time.time()
    if "audio" not in (await request.files):
        return jsonify({"error": "No audio file provided."}), 400
    file = (await request.files)["audio"]
    
    try:
        samples, sample_rate = audio_file_to_samples(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Generate fingerprints using the multi-resolution method.
    query_fps = generate_fingerprints_multiresolution(samples, sample_rate)
    if not query_fps:
        return jsonify({"error": "No fingerprints generated from audio."}), 500

    # Build mapping: fingerprint hash --> list of candidate offsets for the query.
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)
    hash_list = list(query_hashes.keys())
    
    # --- Improved Query: Split hash_list into batches and query concurrently ---
    batch_size = 8  # Tune as needed (try a larger batch size if your hardware can handle it)
    hash_batches = [hash_list[i:i+batch_size] for i in range(0, len(hash_list), batch_size)]
    
    start_find = time.time()
    batch_results = await asyncio.gather(*(find_fingerprint_batch(batch) for batch in hash_batches))
    db_docs = list(chain.from_iterable(batch_results))
    find_query_time = time.time() - start_find

    # Group the DB fingerprints by hash.
    db_group = defaultdict(list)
    for doc in db_docs:
        h = doc["hash"]
        db_group[h].append((doc["song_id"], doc["offset"]))

    bin_width = 0.2  # seconds; adjustable for offset binning
    global_votes = defaultdict(int)

    # Instead of limiting concurrency via a semaphore, run tasks concurrently for every hash.
    async def process_hash(h, query_offsets):
        if h not in db_group:
            return
        query_offsets_array = np.array(query_offsets, dtype=np.float64)
        for song_id, db_offset in db_group[h]:
            votes_for_hash = accumulate_votes_for_hash(query_offsets_array, db_offset, bin_width)
            merge_votes(global_votes, votes_for_hash, song_id)
    
    start_match = time.time()
    # Launch tasks concurrently for all hashes (without semaphore)
    await asyncio.gather(*(process_hash(h, offsets) for h, offsets in query_hashes.items()))
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

    song = await songs_col.find_one({"_id": best_song_id})
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
    # Run the Quart app with an ASGI server (e.g., uvicorn or hypercorn)
    # Example: uvicorn app:app --host 0.0.0.0 --port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
