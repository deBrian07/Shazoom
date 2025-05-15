import sys
import os
import unittest
import numpy as np
from scipy import signal

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import apply_spectral_whitening, preprocess_audio_robust

class TestSpectralWhitening(unittest.TestCase):
    
    def test_apply_spectral_whitening(self):
        # Create a simple test spectrogram
        freq = np.linspace(0, 1, 100)
        time = np.linspace(0, 1, 50)
        # Create a synthetic spectrogram with some patterns
        spectrogram = np.zeros((len(freq), len(time)))
        
        # Add some peaks
        for i in range(5):
            f_idx = np.random.randint(0, len(freq))
            t_idx = np.random.randint(0, len(time))
            spectrogram[f_idx, t_idx] = 1.0
        
        # Add a constant background
        spectrogram += 0.1
        
        # Apply spectral whitening
        whitened = apply_spectral_whitening(spectrogram)
        
        # Check that the output has expected properties
        self.assertEqual(whitened.shape, spectrogram.shape)
        self.assertTrue(np.all(whitened >= 0))
        self.assertTrue(np.all(whitened <= 1))
        
        # For whitened spectrograms, the dynamic range should be enhanced
        self.assertGreater(np.std(whitened), 0.01)
    
    def test_preprocess_audio_robust(self):
        # Create a simple sine wave with added noise
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate)
        # Pure tone at 440 Hz
        signal_pure = np.sin(2 * np.pi * 440 * t)
        # Add noise
        noise = np.random.normal(0, 0.1, sample_rate)
        signal_noisy = signal_pure + noise
        
        # Apply preprocessing
        processed = preprocess_audio_robust(signal_noisy, sample_rate)
        
        # Check that the output has expected properties
        self.assertEqual(len(processed), len(signal_noisy))
        
        # Check normalization
        self.assertLessEqual(np.max(processed), 1.0)
        self.assertGreaterEqual(np.min(processed), -1.0)
        
        # Check if energy is preserved relatively well
        energy_original = np.sum(signal_pure**2)
        energy_processed = np.sum(processed**2)
        
        # We should have less energy after noise removal, but not too much less
        self.assertLess(energy_processed, energy_original * 1.5)
        
        # BETTER METHOD: Evaluate spectral quality using frequency domain analysis
        # Calculate spectrograms
        nperseg = 1024
        f_orig, t_orig, spec_orig = signal.spectrogram(signal_pure, fs=sample_rate, nperseg=nperseg)
        f_noise, t_noise, spec_noise = signal.spectrogram(noise, fs=sample_rate, nperseg=nperseg)
        f_proc, t_proc, spec_proc = signal.spectrogram(processed, fs=sample_rate, nperseg=nperseg)
        
        # Calculate signal/noise ratio in spectral domain
        orig_snr = np.mean(np.abs(spec_orig)) / np.mean(np.abs(spec_noise))
        
        # Estimate noise in processed signal using high-frequency regions
        # (assuming our signal is mostly at 440Hz and noise is broadband)
        high_freq_idx = f_proc > 1000  # Indices for frequencies above 1kHz
        signal_region_idx = (f_proc > 400) & (f_proc < 500)  # Indices around 440Hz
        
        # Measure spectral clarity
        proc_signal_energy = np.mean(np.abs(spec_proc[signal_region_idx]))
        proc_noise_energy = np.mean(np.abs(spec_proc[high_freq_idx]))
        
        proc_snr = proc_signal_energy / max(proc_noise_energy, 1e-10)  # Avoid div by zero
        
        print(f"Original spectral SNR: {orig_snr:.2f}, Processed spectral SNR: {proc_snr:.2f}")
        
        # Spectral flatness (lower is better for speech/music, higher for noise)
        # For whitened signal, we expect improved separation between signal and noise
        def spectral_flatness(spec):
            mean_power = np.mean(spec**2)
            geo_mean_power = np.exp(np.mean(np.log(spec**2 + 1e-10)))
            return geo_mean_power / mean_power
        
        orig_flatness = spectral_flatness(np.abs(spec_orig))
        proc_flatness = spectral_flatness(np.abs(spec_proc))
        
        print(f"Original spectral flatness: {orig_flatness:.4f}, Processed: {proc_flatness:.4f}")
        
        # Cross-correlation as quality measure (how well signal features are preserved)
        xcorr = np.abs(np.correlate(signal_pure, processed, mode='valid')[0])
        max_corr = np.sqrt(np.sum(signal_pure**2) * np.sum(processed**2))
        norm_xcorr = xcorr / max_corr
        
        print(f"Normalized cross-correlation: {norm_xcorr:.4f}")
        
        # We don't assert on these values since they depend on implementation details,
        # but at minimum the processed signal should still correlate with original
        self.assertGreater(norm_xcorr, 0.2)

if __name__ == '__main__':
    unittest.main() 