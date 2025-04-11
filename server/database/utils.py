import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from pydub import AudioSegment

from numba import njit, types
from numba.typed import Dict as TypedDict

# --- Pre-processing functions ---
def low_pass_filter(samples, cutoff, sample_rate):
    """Applies a first‐order low‐pass filter with cutoff frequency (Hz)."""
    rc = 1.0 / (2 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    filtered = np.empty_like(samples)
    filtered[0] = samples[0] * alpha
    for i in range(1, len(samples)):
        filtered[i] = alpha * samples[i] + (1 - alpha) * filtered[i - 1]
    return filtered

def downsample(samples, original_rate, target_rate):
    """Downsamples the samples by averaging groups of samples."""
    ratio = original_rate // target_rate
    return np.array([np.mean(samples[i:i+ratio]) for i in range(0, len(samples), ratio)])


# --- Audio loading ---
def audio_file_to_samples(file_obj):
    """
    Loads an audio file from a file-like object and converts it to mono.
    
    It then applies a low-pass filter with a 5kHz cutoff and downsamples 
    from 44.1 kHz to approximately 11.025 kHz, matching the seek‑tune approach.
    
    Returns:
        samples (numpy array): Filtered and downsampled audio samples (floats).
        sample_rate (int): New sample rate (approximately 11025).
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
    
    # Pre-filter and downsample
    filtered_samples = low_pass_filter(samples, cutoff=5000, sample_rate=44100)
    downsampled_samples = downsample(filtered_samples, original_rate=44100, target_rate=44100 // 4)  # -> ~11025 Hz
    return downsampled_samples, 44100 // 4  # Effective rate

# --- Fingerprint Generation ---
def generate_fingerprints(samples, sample_rate,
                          threshold_multiplier=3,   # Adaptive multiplier (try tuning between 3 and 4)
                          filter_coef=0.5,            # Global filtering coefficient
                          fanout=4,                 # Maximum pairings per candidate (controls target zone size)
                          window_secs=5.0,            # Maximum pairing time gap (sec)
                          window_size=4096,           # FFT window length (samples) (for high frequency resolution)
                          hop_size=1024,              # Hop size (for temporal resolution)
                          band_boundaries=None):
    """
    Generates fingerprints using adaptive candidate extraction and sliding-window pairing.
    
    Changes compared to your old version:
      - The signal is assumed to be downsampled (≈11.025 kHz) from audio_file_to_samples.
      - The spectrogram is computed with a Hamming window.
      - Adaptive thresholding now uses median + (threshold_multiplier * std) per time slice.
      - Candidate extraction is performed in predefined bands in terms of FFT bins.
      
    For the bands, if there are at least 512 bins (up to 5000 Hz) the default is set to:
         [0, 10, 20, 40, 80, 160, 512]
    Otherwise, the boundaries are scaled proportionally.
    
    Returns:
      A list of tuples (hash_str, candidate_time)
      where hash_str has the form: "int(anchor_freq):int(candidate_freq):int(delta_t*100)"
    """
    # Compute spectrogram with a Hamming window.
    hamming_window = np.hamming(window_size)
    freqs, times, spec = signal.spectrogram(
        samples, fs=sample_rate, window=hamming_window,
        nperseg=window_size, noverlap=window_size - hop_size, mode='magnitude'
    )
    spec = np.abs(spec)
    
    # Limit frequencies to below 5000 Hz.
    valid_idx = np.where(freqs < 5000)[0]
    if valid_idx.size == 0:
        return []
    max_bin = valid_idx[-1] + 1
    freqs = freqs[:max_bin]
    spec = spec[:max_bin, :]  # shape: (n_bins, n_times)
    
    # Set band boundaries in terms of FFT bin indices.
    if band_boundaries is None:
        if max_bin >= 512:
            band_boundaries = [0, 10, 20, 40, 80, 160, 512]
        else:
            factor = max_bin / 512.0
            band_boundaries = [int(b * factor) for b in [0, 10, 20, 40, 80, 160, 512]]
    
    # Apply 2D maximum filter to detect local maxima.
    local_max = (spec == maximum_filter(spec, size=(3, 3)))
    n_times = spec.shape[1]
    candidates = []  # List of tuples: (time, frequency, amplitude)
    
    for t_idx in range(n_times):
        spectrum = spec[:, t_idx]
        # Adaptive threshold using median instead of mean improves robustness.
        amp_threshold = np.median(spectrum) + (threshold_multiplier * np.std(spectrum))
        local_peaks = np.where((local_max[:, t_idx]) & (spectrum >= amp_threshold))[0]
        slice_candidates = []
        for b in range(len(band_boundaries) - 1):
            low_bound = band_boundaries[b]
            high_bound = band_boundaries[b + 1]
            band_idx = np.where((np.arange(len(freqs)) >= low_bound) & (np.arange(len(freqs)) < high_bound))[0]
            candidate_idx = np.intersect1d(band_idx, local_peaks)
            if candidate_idx.size == 0:
                continue
            best_idx_local = candidate_idx[np.argmax(spectrum[candidate_idx])]
            slice_candidates.append((times[t_idx], freqs[best_idx_local], spectrum[best_idx_local]))
        candidates.extend(slice_candidates)
    
    if not candidates:
        return []
    
    all_amps = np.array([amp for (_, _, amp) in candidates])
    global_mean = np.mean(all_amps)
    filtered_candidates = [(t, f) for (t, f, amp) in candidates if amp >= (global_mean * filter_coef)]
    if not filtered_candidates:
        return []
    
    filtered_candidates.sort(key=lambda x: x[0])
    
    # Pair candidates in a sliding-window (target zone) approach.
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
                hash_str = f"{int(f1)}:{int(f2)}:{int(dt * 100)}"
                fingerprints.append((hash_str, t1))
                count += 1
    return fingerprints

def generate_fingerprints_multiresolution(samples, sample_rate):
    """
    Generates fingerprints at multiple resolutions and returns the union.
    Each configuration is tagged with a version identifier.
    
    Two configurations are used:
        - "high_freq": longer window (4096) for high frequency resolution.
        - "high_time": shorter window (1024) for higher temporal resolution.
    
    Returns:
      A list of tuples (hash_str, candidate_time)
    """
    configs = [
        (4096, 1024, "high_freq"),
        (1024, 256, "high_time")
    ]
    all_fps = []
    for window_size, hop_size, version in configs:
        fps = generate_fingerprints(
            samples, sample_rate,
            threshold_multiplier=4,  # Tune if necessary
            filter_coef=0.7,         # Tune if necessary
            fanout=7,                # Tune: smaller fanout yields fewer pairings
            window_secs=5.0,
            window_size=window_size,
            hop_size=hop_size,
            band_boundaries=None  # Defaults to our predefined bin boundaries.
        )
        # Append version information to each fingerprint.
        fps_with_version = [(f"{hash_str}:{version}", candidate_time) for (hash_str, candidate_time) in fps]
        all_fps.extend(fps_with_version)
    return all_fps

# --- Numba-accelerated helper ---
@njit
def accumulate_votes_for_hash(query_offsets, db_offset, bin_width):
    """
    For a single hash, given an array of query candidate offsets and a db offset,
    accumulate votes per binned delta. Returns a dict mapping binned_delta (float64) to count (int64).
    (We use numba.typed.Dict as a typed dictionary.)
    """
    votes = TypedDict.empty(key_type=types.float64, value_type=types.int64)
    # For each query offset, compute the delta and its binned value.
    for i in range(query_offsets.shape[0]):
        delta = db_offset - query_offsets[i]
        # Here we use Python's built-in round; in numba, it returns a float.
        binned_delta = round(delta / bin_width) * bin_width
        if binned_delta in votes:
            votes[binned_delta] += 1
        else:
            votes[binned_delta] = 1
    return votes

def merge_votes(global_votes, new_votes, song_id):
    """
    Merges votes from new_votes (a numba typed dict) into the global_votes dictionary.
    The keys of global_votes are tuples (song_id, binned_delta).
    """
    for key in new_votes:
        global_key = (song_id, key)
        global_votes[global_key] = global_votes.get(global_key, 0) + new_votes[key]