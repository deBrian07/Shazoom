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

def audio_file_to_samples(file_path):
    """
    Loads an audio file and converts it to a mono stream at 44.1 kHz.
    
    Returns:
        samples (numpy array): Normalized audio samples.
        sample_rate (int): Fixed at 44100.
    """
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as err:
        raise Exception(f"Error loading audio file '{file_path}': {err}")

    # Convert to mono and resample to 44.1 kHz.
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize 16-bit audio (sample_width == 2) to float values in [-1, 1].
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

def main():
    # Get the MongoDB connection string from the environment variable.
    # MONGO_URI = os.environ.get("MONGO_URI")
    # MONGO_URI = "mongodb+srv://debriann07:LtezVMIT4MlVOKMs@shazoom.trk2vxr.mongodb.net/?retryWrites=true&w=majority&appName=Shazoom"
    # MONGO_URI = "mongodb://deBriann07:qeDqRaFeyUBJHQ4S@ac-obdf1cb-shard-00-00.trk2vxr.mongodb.net:27017,ac-obdf1cb-shard-00-01.trk2vxr.mongodb.net:27017,ac-obdf1cb-shard-00-02.trk2vxr.mongodb.net:27017/?replicaSet=atlas-os8k2z-shard-0&ssl=true&authSource=admin&retryWrites=true&w=majority&appName=Shazoom"
    MONGO_URI = "mongodb://localhost:27017/musicDB"
    if not MONGO_URI:
        print("Error: The MONGO_URI environment variable is not set.")
        return

    # Connect to MongoDB.
    client = MongoClient(MONGO_URI)
    db = client["musicDB"]
    songs_col = db["songs"]
    fingerprints_col = db["fingerprints"]

    # Create an index on the 'hash' field.
    fingerprints_col.create_index("hash")
    
    # Set the path for the CSV file.
    csv_path = os.path.join("..", "download", "processed.csv")
    
    # Read CSV file: it must have headers "song name", "artist", "wav file location"
    songs_to_add = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
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
        
        # Generate fingerprints.
        fingerprints = generate_fingerprints(samples, sr)
        if not fingerprints:
            print(f"Warning: No fingerprints generated for '{song['title']}'.")
            continue
        
        # Insert song metadata.
        song_doc = {"title": song["title"], "artist": song["artist"]}
        result = songs_col.insert_one(song_doc)
        song_id = result.inserted_id
        
        # Create fingerprint documents; each document references the song via song_id.
        fp_docs = [{"song_id": song_id, "hash": hash_val, "offset": offset}
                   for (hash_val, offset) in fingerprints]
        fingerprints_col.insert_many(fp_docs)
        print(f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'.")

if __name__ == "__main__":
    main()
