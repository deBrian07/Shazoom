import os
import csv
import numpy as np
from pymongo import MongoClient
from pydub import AudioSegment
from scipy import signal
from tqdm import tqdm
import concurrent.futures

def audio_file_to_samples(file_path):
    """
    Loads an audio file and converts it to a mono stream at 44.1 kHz.
    
    Returns:
        samples (numpy array): Normalized audio samples as floats.
        sample_rate (int): 44100.
    """
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as err:
        raise Exception(f"Error loading audio file '{file_path}': {err}")
    
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        samples = samples.astype(np.int16) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples, 44100

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

def process_song(song, songs_col, fingerprints_col):
    """
    Processes a single song:
      - Checks for duplicates.
      - Loads audio.
      - Generates fingerprints using the enhanced function.
      - Inserts song metadata and fingerprint documents into MongoDB.
      
    Returns a result message.
    """
    if songs_col.find_one({"title": song["title"], "artist": song["artist"]}):
        return f"Skipping '{song['title']}' by {song['artist']}: Already exists."
    
    file_path = song["file"]
    try:
        samples, sr = audio_file_to_samples(file_path)
    except Exception as e:
        return f"Failed to load audio from {file_path}: {e}"
    
    fingerprints = generate_fingerprints(samples, sr)
    if not fingerprints:
        return f"Warning: No fingerprints generated for '{song['title']}'."
    
    song_doc = {"title": song["title"], "artist": song["artist"]}
    result = songs_col.insert_one(song_doc)
    song_id = result.inserted_id
    
    fp_docs = [{"song_id": song_id, "hash": hash_val, "offset": offset}
               for (hash_val, offset) in fingerprints]
    fingerprints_col.insert_many(fp_docs)
    
    return f"Inserted {len(fp_docs)} fingerprints for '{song['title']}'."

def main():
    # Set MongoDB connection string.
    MONGO_URI = "mongodb://localhost:27017/musicDB"
    client = MongoClient(MONGO_URI)
    db = client["musicDB"]
    songs_col = db["songs"]
    fingerprints_col = db["fingerprints"]
    fingerprints_col.create_index("hash")
    
    # Read CSV file with headers: "song name", "artist", "wav file location".
    csv_path = os.path.join("..", "download", "processed.csv")
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
    # Process songs concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
