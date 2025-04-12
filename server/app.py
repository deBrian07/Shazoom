import os
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from collections import defaultdict, Counter
from database.utils import audio_file_to_samples, generate_fingerprints_multiresolution, accumulate_votes_for_hash, merge_votes

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

@app.route('/identify', methods=['POST'])
def identify_song():
    """
    POST /identify:
    Expects a multipart form-data upload with an 'audio' file.
    Processes the audio to extract multi-resolution fingerprints and performs
    offset alignment voting to identify the song.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files["audio"]
    try:
        samples, sample_rate = audio_file_to_samples(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Use the new multi-resolution fingerprint function.
    query_fps = generate_fingerprints_multiresolution(samples, sample_rate)
    if not query_fps:
        return jsonify({"error": "No fingerprints generated from audio."}), 500

    # Build a mapping: fingerprint hash -> list of query candidate offsets
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)
    
    hash_list = list(query_hashes.keys())
    # Use projection to retrieve only needed fields.
    db_cursor = fingerprints_col.find({"hash": {"$in": hash_list}}, {"hash": 1, "song_id": 1, "offset": 1})
    
    bin_width = 0.2  # seconds; adjustable parameter for offset binning
    global_votes = {}  # Python dictionary: keys are tuples (song_id, binned_delta)
    
    # For each matching fingerprint in DB, call the numba-accelerated function for that hash.
    for db_fp in db_cursor:
        song_id = db_fp["song_id"]
        db_offset = db_fp["offset"]
        h = db_fp["hash"]
        if h in query_hashes:
            # Convert the query offsets list to a numpy array.
            query_offsets_array = np.array(query_hashes[h], dtype=np.float64)
            votes_for_hash = accumulate_votes_for_hash(query_offsets_array, db_offset, bin_width)
            merge_votes(global_votes, votes_for_hash, song_id)

    if not global_votes:
        return jsonify({"result": "No match found."}), 200

    vote_counts = Counter(global_votes)
    best_match, best_votes = vote_counts.most_common(1)[0]
    best_song_id, best_delta = best_match

    MIN_VOTES = 20
    if best_votes < MIN_VOTES:
        return jsonify({"result": "No match found."}), 200

    song = songs_col.find_one({"_id": best_song_id})
    if not song:
        return jsonify({"result": "No match found."}), 200

    total_query = len(query_fps)
    vote_ratio = best_votes / total_query if total_query > 0 else 0

    # Normalization: use total count of fingerprints stored in the DB for this song.
    total_db = fingerprints_col.count_documents({"song_id": best_song_id})
    normalized_score = best_votes / total_db if total_db > 0 else 0

    return jsonify({
        "song": song.get("title"),
        "artist": song.get("artist"),
        "offset": best_delta,
        "raw_votes": best_votes,
        "total_db_fps": total_db,
        "vote_ratio": vote_ratio,
        "normalized_score": normalized_score
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
