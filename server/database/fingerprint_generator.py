import os
import csv
import numpy as np
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal
from tqdm import tqdm
import concurrent.futures

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
    
    # Normalize 16-bit audio to float values in [-1, 1].
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

def process_song(song, songs_col, fingerprints_col):
    """
    Processes a single song:
      - Checks if the song already exists (duplicate prevention).
      - Loads audio.
      - Generates fingerprints using the original function.
      - Inserts song metadata and fingerprints into MongoDB.
    
    Returns a result message.
    """
    # Check for duplicate song based on title and artist.
    if songs_col.find_one({"title": song["title"], "artist": song["artist"]}):
        return f"Skipping '{song['title']}' by {song['artist']}: Already exists."
    
    file_path = song["file"]
    try:
        samples, sr = audio_file_to_samples(file_path)
    except Exception as e:
        return f"Failed to load audio from {file_path}: {e}"
    
    # Generate fingerprints using the original method.
    fingerprints = generate_fingerprints(samples, sr)
    if not fingerprints:
        return f"Warning: No fingerprints generated for '{song['title']}'."
    
    # Insert song metadata.
    song_doc = {"title": song["title"], "artist": song["artist"]}
    result = songs_col.insert_one(song_doc)
    song_id = result.inserted_id
    
    # Prepare and insert fingerprint documents.
    fp_docs = [{"song_id": song_id, "hash": hash_val, "offset": offset} 
               for (hash_val, offset) in fingerprints]
    fingerprints_col.insert_many(fp_docs)
    
    return f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'."

def main():
    # Set your MongoDB connection string.
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
    
    # Read CSV file (with headers: "song name", "artist", "wav file location").
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

    results = []
    # Use ThreadPoolExecutor for concurrent processing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_song, song, songs_col, fingerprints_col)
                   for song in songs_to_add]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Songs"):
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()
