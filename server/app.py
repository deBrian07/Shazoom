import os
import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal

app = Flask(__name__)

# Set MongoDB connection string. Change this or set the environment variable MONGO_URI.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/musicDB")
client = MongoClient(MONGO_URI)
db = client["musicDB"]
songs_col = db["songs"]
fingerprints_col = db["fingerprints"]

# Ensure that there is an index on the "hash" field for faster lookups.
fingerprints_col.create_index("hash")

def generate_fingerprints(samples, sample_rate):
    """
    Generate fingerprint hashes for an audio sample using spectral peak pairing.

    This function computes a spectrogram with a Hann window,
    identifies prominent peaks (using a threshold of 5× the mean amplitude in each slice),
    and then pairs each peak with several future peaks (within a 5-second window)
    to produce a list of fingerprint hashes.

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

    # Find prominent peaks in each time slice.
    for t_idx, spectrum in enumerate(spec_magnitude.T):
        peaks, properties = signal.find_peaks(spectrum, height=np.mean(spectrum) * 5)
        if peaks.size:
            # Take the top 5 highest peaks in this time slice.
            top_peaks = sorted(peaks, key=lambda idx: spectrum[idx], reverse=True)[:5]
            for idx in top_peaks:
                peak_points.append((times[t_idx], freqs[idx]))

    fingerprints = []
    fanout = 5        # Number of future peaks to pair with.
    window_secs = 5.0 # Maximum allowable time difference in seconds.

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
        # pydub.AudioSegment can read from file-like objects.
        audio = AudioSegment.from_file(file_obj)
    except Exception as err:
        raise Exception(f"Error loading audio file: {err}")

    # Convert to mono and set frame rate to 44.1 kHz.
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize 16-bit audio to float in [-1, 1].
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
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

    # Generate fingerprints using the provided algorithm.
    sample_fingerprints = generate_fingerprints(samples, sample_rate)
    if not sample_fingerprints:
        return jsonify({"error": "No fingerprints could be generated from the audio."}), 500

    # For every fingerprint hash, search for matches and count votes for each song.
    match_counts = {}
    for hash_val, offset in sample_fingerprints:
        # Look up matching fingerprints in the MongoDB collection.
        results = fingerprints_col.find({"hash": hash_val})
        for fp in results:
            song_id = fp["song_id"]
            match_counts[song_id] = match_counts.get(song_id, 0) + 1

    if not match_counts:
        return jsonify({"result": "No match found."}), 200

    # Determine the best matching song (highest vote count).
    best_match = max(match_counts, key=match_counts.get)
    song = songs_col.find_one({"_id": best_match})
    if not song:
        return jsonify({"result": "No match found."}), 200

    return jsonify({"song": song.get("title"), "artist": song.get("artist")}), 200

if __name__ == '__main__':
    # Listen on 0.0.0.0:5000; set debug=False in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
