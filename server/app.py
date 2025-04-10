import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB connection string
MONGO_URI = "mongodb://localhost:27017/musicDB"
client = MongoClient(MONGO_URI)
db = client["musicDB"]
songs_col = db["songs"]
fingerprints_col = db["fingerprints"]
fingerprints_col.create_index("hash")

def audio_file_to_samples(file_obj):
    """
    Loads an audio file from a file-like object and converts it to mono at 44.1 kHz.
    
    Returns:
        samples (numpy array): Normalized audio samples.
        sample_rate (int): 44100.
    """
    try:
        audio = AudioSegment.from_file(file_obj)
    except Exception as err:
        raise Exception(f"Error loading audio file: {err}")
    
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

def generate_fingerprints(samples, sample_rate,
                          threshold_multiplier=3,  # Lowered to allow more peaks per time slice
                          filter_coef=0.5,           # Lowered to keep more candidates through
                          fanout=10,                 # Increased to pair more candidates
                          window_secs=5.0,           # Pairing window in seconds
                          window_size=4096,          # FFT window length (samples)
                          hop_size=1024,             # Smaller hop size for higher temporal resolution
                          band_boundaries=None):
    """
    Generates fingerprints using band filtering, 2D local maximum detection, and a sliding-window pairing strategy.
    
    Returns:
      A list of tuples: (hash_str, candidate_time)
    """
    if band_boundaries is None:
        band_boundaries = [0, 500, 1000, 2000, 3000, 4000, 5000]
    
    freqs, times, spec = signal.spectrogram(
        samples, fs=sample_rate, window='hann',
        nperseg=window_size, noverlap=window_size - hop_size
    )
    spec = np.abs(spec)
    
    valid_idx = np.where(freqs < 5000)[0]
    if valid_idx.size == 0:
        return []
    max_bin = valid_idx[-1] + 1
    freqs = freqs[:max_bin]
    spec = spec[:max_bin, :]
    
    local_max = (spec == maximum_filter(spec, size=(3, 3)))
    n_times = spec.shape[1]
    candidates = []
    
    for t_idx in range(n_times):
        spectrum = spec[:, t_idx]
        # Use the mean amplitude times threshold_multiplier as a cutoff.
        amp_threshold = np.mean(spectrum) * threshold_multiplier
        local_peaks = np.where((local_max[:, t_idx]) & (spectrum >= amp_threshold))[0]
        slice_candidates = []
        n_bands = len(band_boundaries) - 1
        for b in range(n_bands):
            low_bound = band_boundaries[b]
            high_bound = band_boundaries[b + 1]
            band_idx = np.where((freqs >= low_bound) & (freqs < high_bound))[0]
            candidate_idx = np.intersect1d(band_idx, local_peaks)
            if candidate_idx.size == 0:
                continue
            best_idx_local = candidate_idx[np.argmax(spectrum[candidate_idx])]
            best_amp = spectrum[best_idx_local]
            candidate_freq = freqs[best_idx_local]
            slice_candidates.append((times[t_idx], candidate_freq, best_amp))
        candidates.extend(slice_candidates)
    
    if not candidates:
        return []
    
    all_amps = np.array([amp for (_, _, amp) in candidates])
    global_mean = np.mean(all_amps)
    filtered_candidates = [(t, f) for (t, f, amp) in candidates if amp >= global_mean * filter_coef]
    if not filtered_candidates:
        return []
    
    filtered_candidates.sort(key=lambda x: x[0])
    
    fingerprints = []
    N = len(filtered_candidates)
    for i in range(N):
        t1, f1 = filtered_candidates[i]
        count = 0
        for j in range(i+1, N):
            t2, f2 = filtered_candidates[j]
            dt = t2 - t1
            if dt > window_secs:
                break
            if count < fanout:
                hash_str = f"{int(f1)}:{int(f2)}:{int(dt*100)}"
                fingerprints.append((hash_str, t1))
                count += 1
    return fingerprints

@app.route('/identify', methods=['POST'])
def identify_song():
    """
    POST /identify:
    Expects a multipart form-data upload with an 'audio' file.
    Processes the audio to extract fingerprints and then performs robust offset alignment voting to identify the song.
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

    # Build a mapping: hash --> list of query offsets.
    query_hashes = defaultdict(list)
    for h, q_offset in query_fps:
        query_hashes[h].append(q_offset)

    # Retrieve matching fingerprints from the DB (bulk query).
    db_cursor = fingerprints_col.find({"hash": {"$in": list(query_hashes.keys())}})

    # Offset alignment voting.
    # For each matching fingerprint in the DB, for each query offset with the same hash,
    # compute delta = (stored_offset - query_offset) and bin that delta by a bin width.
    bin_width = 0.1  # seconds; adjust if needed.
    votes = defaultdict(int)
    for db_fp in db_cursor:
        song_id = db_fp.get("song_id")
        db_offset = db_fp.get("offset")
        h = db_fp["hash"]
        if h in query_hashes:
            for q_offset in query_hashes[h]:
                delta = db_offset - q_offset
                # Bin the delta to reduce noise from small timing differences.
                binned_delta = round(delta / bin_width) * bin_width
                votes[(song_id, binned_delta)] += 1

    if not votes:
        return jsonify({"result": "No match found."}), 200

    vote_counts = Counter(votes)
    best_match, best_votes = vote_counts.most_common(1)[0]
    best_song_id, best_delta = best_match

    # Use a minimum vote threshold to ensure reliability.
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
