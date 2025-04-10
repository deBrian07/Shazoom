import os
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter
from database.utils import audio_file_to_samples, generate_fingerprints 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB connection string
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

DEV_MODE = False
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]
    
db = client["musicDB"]
songs_col = db["songs"]
fingerprints_col = db["fingerprints"]
fingerprints_col.create_index("hash")

@app.route('/identify', methods=['POST'])
def identify_song():
    """
    POST /identify:
    Expects a multipart form-data upload with an 'audio' file.
    Processes the audio using the optimized fingerprint pipeline and performs
    offset alignment voting to identify the song.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files["audio"]
    try:
        samples, sample_rate = audio_file_to_samples(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    query_fps = generate_fingerprints(samples, sample_rate)
    if not query_fps:
        return jsonify({"error": "No fingerprints generated from audio."}), 500

    # Map: fingerprint hash --> list of query candidate offsets.
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)
    
    # Bulk query the DB.
    hash_list = list(query_hashes.keys())
    db_cursor = fingerprints_col.find({"hash": {"$in": hash_list}})
    
    bin_width = 0.1  # seconds; for offset binning
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

    song = songs_col.find_one({"_id": best_song_id})
    if not song:
        return jsonify({"result": "No match found."}), 200

    total_query = len(query_fps)
    match_score = best_votes / total_query

    return jsonify({
        "song": song.get("title"),
        "artist": song.get("artist"),
        "offset": best_delta,
        "score": best_votes,
        "match_score": match_score
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
