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
                          threshold_multiplier=3,   # lower to allow more peaks
                          filter_coef=0.5,            # lower to retain more candidates
                          fanout=10,                  # allow more candidate pairings
                          window_secs=5.0,            # pairing window in seconds
                          window_size=4096,           # FFT window length
                          hop_size=1024,              # smaller hop = higher temporal resolution
                          band_boundaries=None):
    """
    Generates fingerprints using an almost fully vectorized candidate extraction and pairing.
    
    Process:
      1. Compute the spectrogram using a Hann window.
      2. Limit frequencies to below 5000 Hz.
      3. For each of the predefined frequency bands, compute in one shot the maximum amplitude per time slice.
      4. Concatenate these candidates across bands into flat arrays of candidate times, frequencies, and amplitudes.
      5. Compute a global mean amplitude over all candidates and keep only those with amplitude >= (global_mean * filter_coef).
      6. Sort the surviving candidates by time.
      7. Pair candidates vectorized: using np.triu_indices, generate pairs (i, j) (with j > i) whose time difference is <= window_secs and index difference <= fanout.
          Build a hash string from int(f1), int(f2), and int(dt * 100).
    
    Returns:
      A list of tuples (hash_str, candidate_time)
    """
    # Default band boundaries: define bands in Hz for the range 0-5000.
    if band_boundaries is None:
        band_boundaries = [0, 500, 1000, 2000, 3000, 4000, 5000]
    
    # Compute spectrogram.
    freqs, times, spec = signal.spectrogram(
        samples, fs=sample_rate, window='hann',
        nperseg=window_size, noverlap=window_size - hop_size
    )
    spec = np.abs(spec)
    
    # Limit frequencies to below 5000 Hz.
    valid_idx = np.where(freqs < 5000)[0]
    if valid_idx.size == 0:
        return []
    max_bin = valid_idx[-1] + 1
    freqs = freqs[:max_bin]
    spec = spec[:max_bin, :]  # shape: (n_bins, n_times)
    
    n_times = spec.shape[1]
    n_bands = len(band_boundaries) - 1
    
    # For each band, extract candidate per time slice in a vectorized manner.
    candidate_times_list = []
    candidate_freqs_list = []
    candidate_amps_list = []
    for b in range(n_bands):
        low_bound = band_boundaries[b]
        high_bound = band_boundaries[b+1]
        # Get indices for bins in this band.
        band_mask = (freqs >= low_bound) & (freqs < high_bound)
        band_indices = np.where(band_mask)[0]
        if band_indices.size == 0:
            continue
        # For each time slice (axis 1), compute maximum amplitude and its index in the band.
        band_spec = spec[band_indices, :]  # shape: (n_band_bins, n_times)
        # Maximum amplitude per column.
        candidate_amps = np.max(band_spec, axis=0)  # shape: (n_times,)
        candidate_idx = np.argmax(band_spec, axis=0)  # relative indices in band_spec
        candidate_freqs = freqs[band_indices][candidate_idx]  # shape: (n_times,)
        candidate_times_list.append(times)      # same times array per band
        candidate_freqs_list.append(candidate_freqs)
        candidate_amps_list.append(candidate_amps)
    
    if not candidate_times_list:
        return []
    # Flatten the candidate arrays: each array now has shape (n_times * n_bands,)
    cand_times = np.concatenate(candidate_times_list)
    cand_freqs = np.concatenate(candidate_freqs_list)
    cand_amps = np.concatenate(candidate_amps_list)
    
    # Global filtering: compute global mean amplitude and filter.
    global_mean = np.mean(cand_amps)
    valid_mask = cand_amps >= (global_mean * filter_coef)
    filtered_times = cand_times[valid_mask]
    filtered_freqs = cand_freqs[valid_mask]
    if filtered_times.size == 0:
        return []
    
    # Sort candidates by time.
    sort_idx = np.argsort(filtered_times)
    sorted_times = filtered_times[sort_idx]
    sorted_freqs = filtered_freqs[sort_idx]
    
    # Pair candidates vectorized.
    N = sorted_times.shape[0]
    if N < 2:
        return []
    # Get indices for all pairs (i, j) with i < j.
    i_idx, j_idx = np.triu_indices(N, k=1)
    dt = sorted_times[j_idx] - sorted_times[i_idx]
    valid_pair_mask = (dt > 0) & (dt <= window_secs) & ((j_idx - i_idx) <= fanout)
    valid_i = i_idx[valid_pair_mask]
    valid_j = j_idx[valid_pair_mask]
    dt_valid = dt[valid_pair_mask]
    
    # Build hash strings.
    f1_int = sorted_freqs[valid_i].astype(int)
    f2_int = sorted_freqs[valid_j].astype(int)
    dt_int = (dt_valid * 100).astype(int)  # time difference in centiseconds
    
    hash_strs = [f"{a}:{b}:{c}" for a, b, c in zip(f1_int, f2_int, dt_int)]
    offsets = sorted_times[valid_i].tolist()
    
    fingerprints = list(zip(hash_strs, offsets))
    return fingerprints

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
