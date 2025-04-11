import os
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from collections import defaultdict, Counter
from database.utils import audio_file_to_samples, generate_fingerprints_multiresolution

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB connection string
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

DEV_MODE = True  # Change to True for testing if needed.
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

    # Map: fingerprint hash --> list of query candidate offsets.
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)
    
    # Bulk query the DB for matching fingerprints.
    hash_list = list(query_hashes.keys())
    db_cursor = fingerprints_col.find({"hash": {"$in": hash_list}})
    
    bin_width = 0.2  # seconds; for offset binning (you may tune this)
    votes = defaultdict(int)
    for db_fp in db_cursor:
        song_id = db_fp.get("song_id")
        db_offset = db_fp.get("offset")
        h = db_fp["hash"]
        if h in query_hashes:
            for q_offset in query_hashes[h]:
                delta = db_offset - q_offset
                binned_delta = round(delta / bin_width) * bin_width
                votes[(song_id, binned_delta)] += 1

    if not votes:
        return jsonify({"result": "No match found."}), 200

    vote_counts = Counter(votes)
    best_match, best_votes = vote_counts.most_common(1)[0]
    best_song_id, best_delta = best_match

    MIN_VOTES = 3
    if best_votes < MIN_VOTES:
        return jsonify({"result": "No match found."}), 200

    # --- Normalization Part ---
    # Retrieve the total number of fingerprints in the database for the best matching song.
    total_db = fingerprints_col.count_documents({"song_id": best_song_id})
    # Compute a normalized score (you might combine this with the query fingerprint count).
    normalized_score = best_votes / total_db if total_db > 0 else 0

    # You might also combine with the query fingerprint count:
    total_query = len(query_fps)
    vote_ratio = best_votes / total_query if total_query > 0 else 0

    song = songs_col.find_one({"_id": best_song_id})
    if not song:
        return jsonify({"result": "No match found."}), 200

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
