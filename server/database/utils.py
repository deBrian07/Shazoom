import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from pydub import AudioSegment

def audio_file_to_samples(file_obj):
    """
    Loads an audio file from a file-like object and converts it to a mono stream at 44.1 kHz.
    
    Returns:
        samples (numpy array): Normalized audio samples as floats.
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
    Generates fingerprints using band filtering, 2D local maximum detection,
    and sliding-window pairing.
    
    Process:
      1. Compute the spectrogram (using a Hann window).
      2. Limit frequencies to below 5000 Hz.
      3. Apply a 2D maximum filter to detect local maxima.
      4. For each time slice, divide the frequency bins into bands 
         (default: [0, 500, 1000, 2000, 3000, 4000, 5000] Hz).
      5. In each band, from candidate bins (local maxima) that exceed an amplitude threshold 
         (mean * threshold_multiplier), select the bin with maximum amplitude.
      6. Collect all candidate peaks (time, frequency, amplitude) per time slice.
      7. Compute the global mean amplitude and filter out candidates with amplitude below (global_mean * filter_coef).
      8. Sort the surviving candidates by time.
      9. For each candidate, pair it with up to 'fanout' subsequent candidates (where time difference ≤ window_secs)
         to form fingerprint hashes of the form:
              "int(anchor_freq):int(candidate_freq):int(delta_t*100)"
    
    Returns:
      A list of tuples (hash_str, candidate_time)
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
    spec = spec[:max_bin, :]  # shape: (n_bins, n_times)
    
    # Apply 2D local maximum filtering.
    local_max = (spec == maximum_filter(spec, size=(3, 3)))
    n_times = spec.shape[1]
    candidates = []  # Will store tuples: (time, frequency, amplitude)
    
    # Process each time slice.
    for t_idx in range(n_times):
        spectrum = spec[:, t_idx]
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
            slice_candidates.append((times[t_idx], freqs[best_idx_local], spectrum[best_idx_local]))
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
                hash_str = f"{int(f1)}:{int(f2)}:{int(dt * 100)}"
                fingerprints.append((hash_str, t1))
                count += 1
    return fingerprints
