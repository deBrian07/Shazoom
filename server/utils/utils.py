import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from pydub import AudioSegment

from numba import njit, types
from numba.typed import Dict as TypedDict
from utils.constants import (
    FANOUT, FILTER_COEF, FINGERPRINT_CONFIGS, THRESHOLD_MULTIPLIER, WINDOW_SECS,
    SPECTRAL_WHITENING_ALPHA, SPECTRAL_WHITENING_BETA, 
    SPECTRAL_WHITENING_PRESERVE_PEAKS, SPECTRAL_WHITENING_DEEMPHASIS
)

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

def apply_spectral_whitening(spectrogram, alpha=0.95, beta=0.7, preserve_peaks=True, deemphasis=True):
    """
    Apply spectral whitening to improve fingerprint robustness against noise.
    
    This implements a technique similar to Shazam's aggressive noise-robust 
    normalization that equalizes amplitude and suppresses background hiss.
    
    Args:
        spectrogram: Input spectrogram (2D numpy array)
        alpha: Smoothing factor for spectral envelope estimation (0.95 is good for music)
        beta: Whitening strength (0=none, 1=full whitening)
        preserve_peaks: Whether to preserve spectral peaks during whitening
        deemphasis: Whether to apply a de-emphasis filter to attenuate high frequencies
        
    Returns:
        Whitened spectrogram with improved SNR
    """
    # Make a copy to avoid modifying the input
    spec = np.copy(spectrogram)
    
    # First, apply a noise floor to avoid boosting near-zero values too much
    noise_floor = np.mean(spec) * 0.01
    spec = np.maximum(spec, noise_floor)
    
    # Find spectral peaks if requested
    peak_mask = None
    if preserve_peaks:
        # Identify spectral peaks to preserve (local maxima)
        peak_mask = (spec == maximum_filter(spec, size=(3, 3)))
    
    # Calculate running spectral envelope
    envelope = np.zeros_like(spec)
    for i in range(spec.shape[1]):
        if i == 0:
            envelope[:, i] = spec[:, i]
        else:
            envelope[:, i] = alpha * envelope[:, i-1] + (1-alpha) * np.maximum(spec[:, i], envelope[:, i-1])
    
    # Apply whitening (normalize by envelope)
    whitened = np.zeros_like(spec)
    for i in range(spec.shape[1]):
        # Avoid division by zero
        mask = envelope[:, i] > 1e-10
        whitened[mask, i] = spec[mask, i] / (envelope[mask, i] ** beta)
    
    # Preserve peaks if requested
    if preserve_peaks and peak_mask is not None:
        # Keep original values at peak locations
        whitened[peak_mask] = spec[peak_mask]
    
    # Apply spectral de-emphasis if requested (attenuate high frequencies)
    if deemphasis:
        # Apply a gradual rolloff for high frequencies
        # Assuming rows represent frequencies from low to high
        num_freqs = whitened.shape[0]
        
        # Create a de-emphasis filter that attenuates higher frequencies
        # Starting at 50% of Nyquist, roll off progressively
        deemph_curve = np.ones(num_freqs)
        midpoint = num_freqs // 2
        
        # Gentle frequency rolloff to attenuate high frequencies
        # This helps match natural speech/music spectrum and reduces artifacts
        if midpoint > 0:
            deemph_curve[midpoint:] = np.exp(-0.5 * np.linspace(0, 4, num_freqs - midpoint))
            
            # Apply the de-emphasis to the whitened spectrogram
            for i in range(whitened.shape[1]):
                whitened[:, i] *= deemph_curve
    
    # Dynamic range compression (logarithmic scaling)
    # Using log1p to avoid log(0) issues and for better perceptual scaling
    compressed = np.log1p(whitened * 10)
    
    # Final normalization to [0,1] range
    min_val = np.min(compressed)
    max_val = np.max(compressed)
    if max_val > min_val:  # Avoid division by zero
        normalized = (compressed - min_val) / (max_val - min_val)
    else:
        normalized = compressed
    
    return normalized

def preprocess_audio_robust(samples, sample_rate):
    """
    Apply noise-robust preprocessing before fingerprinting.
    
    This implements Shazam-like preprocessing for improved recognition accuracy
    in noisy environments.
    
    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Preprocessed samples ready for fingerprinting
    """
    # Apply bandpass filter (keep 100-5000Hz where most music info exists)
    nyquist = sample_rate / 2
    low, high = 100 / nyquist, 5000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, samples)
    
    # Calculate spectrogram
    nperseg = int(0.025 * sample_rate)  # 25ms windows
    noverlap = int(nperseg * 0.5)       # 50% overlap
    
    # Use Hann window for better frequency resolution
    f, t, spec = signal.spectrogram(filtered, fs=sample_rate, 
                                  nperseg=nperseg, noverlap=noverlap,
                                  window='hann', mode='complex')
    
    # Apply spectral whitening to the magnitude of the spectrogram
    mag_spec = np.abs(spec)
    whitened_mag = apply_spectral_whitening(mag_spec, 
                                           alpha=SPECTRAL_WHITENING_ALPHA,
                                           beta=SPECTRAL_WHITENING_BETA,
                                           preserve_peaks=SPECTRAL_WHITENING_PRESERVE_PEAKS,
                                           deemphasis=SPECTRAL_WHITENING_DEEMPHASIS)
    
    # Phase preservation is critical for audio quality
    # Use the original phase information
    phase = np.angle(spec)
    
    # Reconstruct complex spectrum
    S_whitened = whitened_mag * np.exp(1j * phase)
    
    # Reconstruct time domain signal with optimal parameters
    _, enhanced = signal.istft(S_whitened, fs=sample_rate, 
                             nperseg=nperseg, noverlap=noverlap, 
                             window='hann')
    
    # Ensure we have an appropriate length
    if len(enhanced) > len(samples):
        enhanced = enhanced[:len(samples)]
    elif len(enhanced) < len(samples):
        enhanced = np.pad(enhanced, (0, len(samples) - len(enhanced)), 'constant')
    
    # Apply post-whitening smoothing filter to reduce potential artifacts
    b, a = signal.butter(2, 0.9)  # Light lowpass to smooth any artifacts
    enhanced = signal.filtfilt(b, a, enhanced)
    
    # Normalize amplitude
    if np.max(np.abs(enhanced)) > 0:
        enhanced = enhanced / np.max(np.abs(enhanced))
    
    return enhanced

def audio_file_to_samples(file_obj):
    """
    Loads an audio file from a file-like object and converts it to mono.
    
    It then applies a low-pass filter with a 5kHz cutoff and downsamples 
    from 44.1 kHz to approximately 11.025 kHz, matching the seek‑tune approach.
    
    Additionally, it applies Shazam-like spectral whitening and noise-robust
    normalization for improved recognition in noisy environments.
    
    Returns:
        samples (numpy array): Filtered, whitened and downsampled audio samples (floats).
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
    
    # Apply initial low-pass filtering
    filtered_samples = low_pass_filter(samples, cutoff=5000, sample_rate=44100)
    
    # Apply the robust spectral whitening preprocessing
    robust_samples = preprocess_audio_robust(filtered_samples, 44100)
    
    # Downsample after preprocessing for efficiency
    downsampled_samples = downsample(robust_samples, original_rate=44100, target_rate=44100 // 4)  # -> ~11025 Hz
    
    return downsampled_samples, 44100 // 4  # Effective rate

def generate_fingerprints(samples, sample_rate,
                          threshold_multiplier=3,   # Adaptive multiplier (try tuning between 3 and 4)
                          filter_coef=0.5,            # Global filtering coefficient
                          fanout=4,                 # Maximum pairings per candidate (controls target zone size)
                          window_secs=5.0,            # Maximum pairing time gap (sec)
                          window_size=4096,           # FFT window length (samples) (for high frequency resolution)
                          hop_size=1024,              # Hop size (for temporal resolution)
                          band_boundaries=None):
    
    # spectrogram with a Hamming window.
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
    spec = spec[:max_bin, :] 
    
    # Set band boundaries in terms of FFT bin indices.
    if band_boundaries is None:
        if max_bin >= 512:
            band_boundaries = [0, 10, 20, 40, 80, 160, 512]
        else:
            factor = max_bin / 512.0
            band_boundaries = [int(b * factor) for b in [0, 10, 20, 40, 80, 160, 512]]
    
    # 2D maximum filter
    local_max = (spec == maximum_filter(spec, size=(3, 3)))
    n_times = spec.shape[1]
    candidates = [] 
    
    for t_idx in range(n_times):
        spectrum = spec[:, t_idx]
        # Adaptive threshold 
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
    configs = FINGERPRINT_CONFIGS
    all_fps = []
    for window_size, hop_size, version in configs:
        fps = generate_fingerprints(
            samples, sample_rate,
            threshold_multiplier=THRESHOLD_MULTIPLIER,  # Tune 
            filter_coef=FILTER_COEF,         # Tune 
            fanout=FANOUT,                # Tune: smaller fanout yields fewer pairings
            window_secs=WINDOW_SECS,
            window_size=window_size,
            hop_size=hop_size,
            band_boundaries=None  
        )
        fps_with_version = [(f"{hash_str}:{version}", candidate_time) for (hash_str, candidate_time) in fps]
        all_fps.extend(fps_with_version)
    return all_fps

# --- Numba-accelerated helper ---
@njit
def accumulate_votes_for_hash(query_offsets, db_offset, bin_width):
    votes = TypedDict.empty(key_type=types.float64, value_type=types.int64)
    for i in range(query_offsets.shape[0]):
        delta = db_offset - query_offsets[i]
        binned_delta = round(delta / bin_width) * bin_width
        if binned_delta in votes:
            votes[binned_delta] += 1
        else:
            votes[binned_delta] = 1
    return votes

def merge_votes(global_votes, new_votes, song_id):
    for key in new_votes:
        global_key = (song_id, key)
        global_votes[global_key] = global_votes.get(global_key, 0) + new_votes[key]

def accumulate_votes_vectorized(query_offsets, docs, bin_width):
    """
    Vectorized vote tally for one hash:
      - query_offsets: 1D numpy array of offsets for this hash
      - docs: list of (song_id, db_offset) tuples
      - bin_width: width of each time‑delta bin
    Returns:
      dict: {(song_id, delta): count}
    """
    # Determine workload size
    total_pairs = len(docs) * len(query_offsets)
    votes = {}

    if total_pairs < 1000:
        # Small case: use existing Numba helper for each doc
        for song_id, db_offset in docs:
            sub = accumulate_votes_for_hash(query_offsets, db_offset, bin_width)
            for delta, cnt in sub.items():
                key = (song_id, delta)
                votes[key] = votes.get(key, 0) + cnt
    else:
        # Large case: vectorized across all docs and offsets
        # Map song_ids to integer indices for array operations
        unique_sids = []
        sid_to_idx = {}
        for sid, _ in docs:
            if sid not in sid_to_idx:
                sid_to_idx[sid] = len(unique_sids)
                unique_sids.append(sid)
        song_idx_arr = np.array([sid_to_idx[sid] for sid, _ in docs], dtype=np.int64)
        db_offsets = np.array([offset for _, offset in docs], dtype=np.float64)

        # Compute pairwise deltas and quantize
        deltas = db_offsets[:, None] - query_offsets[None, :]
        binned = np.round(deltas / bin_width) * bin_width

        # Flatten for counting
        flat_idxs = np.repeat(song_idx_arr, len(query_offsets))
        flat_deltas = binned.ravel()

        # Structured array to count unique pairs
        pairs = np.empty(flat_idxs.shape[0], dtype=[('idx', 'i8'), ('delta', 'f8')])
        pairs['idx'] = flat_idxs
        pairs['delta'] = flat_deltas
        uniq, counts = np.unique(pairs, return_counts=True)

        # Build result dict mapping back to original song_ids
        for entry, cnt in zip(uniq, counts):
            sid = unique_sids[int(entry['idx'])]
            delta = float(entry['delta'])
            key = (sid, delta)
            votes[key] = cnt

    return votes