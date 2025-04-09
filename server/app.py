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

def generate_fingerprints(samples, sample_rate):
    """
    Generate fingerprint hashes for an audio sample using spectral peak pairing.
    
    This simplified version:
      - Computes a spectrogram with a Hann window.
      - Identifies prominent peaks per time slice (using a threshold of 5× the mean amplitude).
      - Pairs each peak with several future peaks (within a 5-second window) to form a hash.
    
    Returns:
        A list of tuples (hash_str, time_offset).
    """
    window_size = 4096  # Number of samples per FFT
    hop_size = 2048     # 50% overlap

    freqs, times, spec = signal.spectrogram(
        samples, fs=sample_rate, window='hann',
        nperseg=window_size, noverlap=window_size - hop_size
    )
    spec_magnitude = np.abs(spec)
    peak_points = []  # List of (time, frequency) tuples

    # Find peaks in each time slice.
    for t_idx, spectrum in enumerate(spec_magnitude.T):
        peaks, properties = signal.find_peaks(spectrum, height=np.mean(spectrum) * 5)
        if peaks.size:
            # Take the top 5 highest peaks
            top_peaks = sorted(peaks, key=lambda idx: spectrum[idx], reverse=True)[:5]
            for idx in top_peaks:
                peak_points.append((times[t_idx], freqs[idx]))

    fingerprints = []
    fanout = 5        # Number of future peaks to pair with
    window_secs = 5.0 # Maximum allowable time difference in seconds

    for i in range(len(peak_points)):
        t1, f1 = peak_points[i]
        for j in range(1, fanout + 1):
            if i + j < len(peak_points):
                t2, f2 = peak_points[i + j]
                if 0 < t2 - t1 <= window_secs:
                    f1_int = int(f1)
                    f2_int = int(f2)
                    delta_t = int((t2 - t1) * 100)  # quantize delta time in centiseconds
                    hash_str = f"{f1_int}:{f2_int}:{delta_t}"
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
