import os
import csv
import numpy as np
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal

def generate_fingerprints(samples, sample_rate):
    """
    Generate fingerprint hashes for an audio sample using spectral peak pairing.
    
    This simplified version:
      - Computes a spectrogram with a Hann window.
      - Identifies prominent peaks for each time slice (using a threshold based on mean amplitude).
      - Pairs each peak with several future peaks (within a time window) to form a hash.
    
    Returns:
        A list of tuples (hash_str, time_offset).
    """
    window_size = 4096  # Number of samples per FFT
    hop_size = 2048     # Overlap amount (50% overlap)
    
    freqs, times, spec = signal.spectrogram(
        samples, fs=sample_rate, window='hann',
        nperseg=window_size, noverlap=window_size - hop_size
    )
    spec_magnitude = np.abs(spec)
    peak_points = []  # List of (time, frequency) tuples

    # For each time slice, find peaks that are significantly above average amplitude.
    for t_idx, spectrum in enumerate(spec_magnitude.T):
        peaks, _ = signal.find_peaks(spectrum, height=np.mean(spectrum) * 5)
        if peaks.size:
            # Take the top 5 highest peaks
            top_peaks = sorted(peaks, key=lambda idx: spectrum[idx], reverse=True)[:5]
            for idx in top_peaks:
                peak_points.append((times[t_idx], freqs[idx]))
    
    fingerprints = []
    fanout = 5        # Number of future peaks to pair with
    window_secs = 5.0 # Maximum allowable time difference between paired peaks
    
    for i in range(len(peak_points)):
        t1, f1 = peak_points[i]
        for j in range(1, fanout + 1):
            if i + j < len(peak_points):
                t2, f2 = peak_points[i + j]
                if 0 < t2 - t1 <= window_secs:
                    # Quantize frequency and time difference to reduce minor variations
                    f1_int = int(f1)
                    f2_int = int(f2)
                    delta_t = int((t2 - t1) * 100)  # quantized in centiseconds
                    hash_str = f"{f1_int}:{f2_int}:{delta_t}"
                    fingerprints.append((hash_str, t1))
    return fingerprints

def audio_file_to_samples(file_path):
    """
    Loads an audio file and converts it to a mono stream at 44.1 kHz.
    
    Returns:
        samples (numpy array): normalized audio samples.
        sample_rate (int): fixed at 44100.
    """
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as err:
        raise Exception(f"Error loading audio file '{file_path}': {err}")
    
    # Convert audio to mono and resample to 44.1 kHz.
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize 16-bit audio to float values between -1 and 1.
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

def main():
    # Configure MongoDB connection. Use the environment variable MONGO_URI if set; otherwise, default locally.
    MONGO_URI = os.environ.get("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["musicDB"]
    songs_col = db["songs"]
    fingerprints_col = db["fingerprints"]

    # Ensure an index on the 'hash' field for fast lookups.
    fingerprints_col.create_index("hash")
    
    # Path to the CSV file containing song metadata.
    csv_path = os.path.join("..", "download", "processed.csv")
    
    # Read CSV file which should have headers: "song name", "artist", "wav file location"
    songs_to_add = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Strip extra spaces and add to the list.
                songs_to_add.append({
                    "title": row["song name"].strip(),
                    "artist": row["artist"].strip(),
                    "file": row["wav file location"].strip()
                })
    except Exception as e:
        print(f"Failed to read CSV file at '{csv_path}': {e}")
        return
    
    # Process each song from the CSV.
    for song in songs_to_add:
        file_path = song["file"]
        print(f"Processing '{song['title']}' from file: {file_path}...")
        try:
            samples, sr = audio_file_to_samples(file_path)
        except Exception as e:
            print(f"Failed to load audio from {file_path}: {e}")
            continue
        
        # Generate fingerprints for the audio sample.
        fingerprints = generate_fingerprints(samples, sr)
        if not fingerprints:
            print(f"Warning: No fingerprints generated for '{song['title']}'.")
            continue
        
        # Insert song metadata into the 'songs' collection.
        song_doc = {"title": song["title"], "artist": song["artist"]}
        result = songs_col.insert_one(song_doc)
        song_id = result.inserted_id
        
        # Insert associated fingerprint entries (each fingerprint references the song via song_id).
        fp_docs = [
            {"song_id": song_id, "hash": hash_val, "offset": offset}
            for (hash_val, offset) in fingerprints
        ]
        fingerprints_col.insert_many(fp_docs)
        print(f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'.")

if __name__ == "__main__":
    main()
