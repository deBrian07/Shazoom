import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set MongoDB connection string. Change this or set the environment variable MONGO_URI.
MONGO_URI = "mongodb://localhost:27017/musicDB"
client = MongoClient(MONGO_URI)
db = client["musicDB"]
songs_col = db["songs"]
fingerprints_col = db["fingerprints"]

# Ensure that there is an index on the "hash" field for faster lookups.
fingerprints_col.create_index("hash")

def generate_fingerprints(samples, sample_rate,
                          threshold_multiplier=5.0,  # Multiplier for adaptive threshold per band
                          filter_coef=1.0,           # Candidate must have amplitude >= global_mean * filter_coef
                          fanout=5,                  # Number of subsequent candidates to pair with
                          window_secs=5.0,           # Maximum allowed time difference (sec) for pairing
                          window_size=4096,          # FFT window length in samples
                          hop_size=2048,             # 50% overlap
                          band_boundaries=None):
    """
    Enhanced fingerprint generation that implements the filtering procedure described in the paper.
    
    Process per song:
      1. Compute the spectrogram (using a Hann window) of the audio samples.
      2. Limit frequencies to below 5000 Hz.
      3. For each time slice, divide the FFT bins into six logarithmic bands.
         (Default bands, in Hz, are defined as follows, but can be overridden:)
           - Very low: 0–500 Hz
           - Low:      500–1000 Hz
           - Low-mid:  1000–2000 Hz
           - Mid:      2000–3000 Hz
           - Mid-high: 3000–4000 Hz
           - High:     4000–5000 Hz
      4. In each band of a time slice, select the candidate corresponding to the bin with maximum amplitude.
      5. Gather these candidate peaks (with their amplitude) for every time slice.
      6. Compute the global mean amplitude of all candidates (from the full song).
      7. For each candidate, only keep it if its amplitude is >= global_mean * filter_coef.
      8. Sort the surviving candidates by time.
      9. Pair each candidate with up to 'fanout' subsequent candidates (if the time difference ≤ window_secs) to form hashes.
         The hash for a pair is the string: "int(f1):int(f2):int(delta_t)", where delta_t is the time difference in centiseconds.
    
    Returns:
      A list of tuples (hash_str, time_offset)
    """
    # Use default band boundaries if none provided.
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
    candidates = []  # Will store tuples (time, frequency, amplitude)
    
    # Process each time slice.
    for t_idx in range(n_times):
        spectrum = spec[:, t_idx]
        slice_candidates = []
        n_bands = len(band_boundaries) - 1
        for b in range(n_bands):
            low_bound = band_boundaries[b]
            high_bound = band_boundaries[b + 1]
            band_idx = np.where((freqs >= low_bound) & (freqs < high_bound))[0]
            if band_idx.size == 0:
                continue
            band_values = spectrum[band_idx]
            best_idx_local = np.argmax(band_values)
            best_amp = band_values[best_idx_local]
            candidate_freq = freqs[band_idx[best_idx_local]]
            slice_candidates.append((times[t_idx], candidate_freq, best_amp))
        candidates.extend(slice_candidates)
    
    if not candidates:
        return []
    
    all_amps = np.array([amp for (_, _, amp) in candidates])
    global_mean = np.mean(all_amps)
    
    # Filter candidates based on amplitude threshold.
    filtered_candidates = [(t, f) for (t, f, amp) in candidates if amp >= global_mean * filter_coef]
    
    # Sort by time.
    filtered_candidates.sort(key=lambda x: x[0])
    
    # Pair candidates using a simple loop (to avoid high memory usage).
    fingerprints = []
    N = len(filtered_candidates)
    for i in range(N):
        t1, f1 = filtered_candidates[i]
        for j in range(1, fanout + 1):
            if i + j < N:
                t2, f2 = filtered_candidates[i + j]
                dt = t2 - t1
                if 0 < dt <= window_secs:
                    f1_int = int(f1)
                    f2_int = int(f2)
                    dt_int = int(dt * 100)  # Convert to centiseconds.
                    hash_str = f"{f1_int}:{f2_int}:{dt_int}"
                    fingerprints.append((hash_str, t1))
    return fingerprints

def audio_file_to_samples(file_obj):
    """
    Loads an audio file from a file-like object and converts it to a mono audio stream at 44.1 kHz.
    Returns:
        samples (numpy array): Normalized audio samples.
        sample_rate (int): Fixed at 44100.
    """
    try:
        audio = AudioSegment.from_file(file_obj)
    except Exception as err:
        raise Exception(f"Error loading audio file: {err}")

    audio = audio.set_channels(1).set_frame_rate(44100)  # Convert to mono, 44.1 kHz
    samples = np.array(audio.get_array_of_samples())
    
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0  # Normalize to [-1, 1]
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

@app.route('/identify', methods=['POST'])
def identify_song():
    """
    POST /identify:
    Expects a multipart form-data upload with an 'audio' file.
    Processes the audio, extracts fingerprints, looks up matching fingerprints in MongoDB,
    and returns the matching song's title and artist, or a "No match" message.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]

    try:
        samples, sample_rate = audio_file_to_samples(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sample_fingerprints = generate_fingerprints(samples, sample_rate)
    if not sample_fingerprints:
        return jsonify({"error": "No fingerprints could be generated from the audio."}), 500

    # === Optimization Start ===
    # Instead of performing a separate DB query for each fingerprint, collect all
    # hash strings and perform a single bulk query using the $in operator.
    hash_list = [hash_val for hash_val, offset in sample_fingerprints]
    cursor = fingerprints_col.find({"hash": {"$in": hash_list}})
    
    match_counts = {}
    for fp in cursor:
        song_id = fp.get("song_id")
        if song_id is not None:
            match_counts[song_id] = match_counts.get(song_id, 0) + 1
    # === Optimization End ===

    if not match_counts:
        return jsonify({"result": "No match found."}), 200

    best_match = max(match_counts, key=match_counts.get)
    song = songs_col.find_one({"_id": best_match})
    if not song:
        return jsonify({"result": "No match found."}), 200

    return jsonify({"song": song.get("title"), "artist": song.get("artist")}), 200

if __name__ == '__main__':
    # Listen on 0.0.0.0:5000 so that external devices can connect.
    app.run(host="0.0.0.0", port=5000, debug=False)
