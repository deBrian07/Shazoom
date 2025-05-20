# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://debrian07.github.io",
    "https://www.shazoom.app",
    "https://api.shazoom.app"
]

DEV_MODE = True  # True when testing something (change to False before committing)

# Fingerprint extraction parameters
# Each tuple: (window_size, hop_size, version_label)
FINGERPRINT_CONFIGS = [
    (4096, 1024, "high_freq"),
    (1024, 256,  "high_time"),
]
THRESHOLD_MULTIPLIER = 3.5  # adaptive peak threshold multiplier
FILTER_COEF = 0.7          # global amplitude filter coefficient
FANOUT = 10                 # number of target pairings per anchor
WINDOW_SECS = 5.0          # max pairing time gap in seconds

# Matching and recording thresholds
MIN_VOTES = 50              # min vote count to accept a match (already tuned higher)
BIN_WIDTH = 0.2             # time-delta quantization width (seconds)
MAX_RECORDING = 9.0         # max recording duration (seconds)
THRESHOLD_TIME = 5.0        # initial wait time before matching (seconds)
MIN_HASHES = 100  
VOTE_MARGIN = 1.5  
TOP_K_NORMALIZE     = 3        # only normalize the top-3 candidates
TOTAL_THRESHOLD_FACTOR = 1.2   # tighten normalized threshold slightly           

# System parameters
RAM_THRESHOLD_BYTES = 58 * 1024 ** 3  # 58 GB, used in RAM-monitoring

TO_PREWARM = 5000  # reduce if memory is tight
BATCH_SIZE = 1000

WORKERS = 8

# define sliding window length in seconds
SLIDING_WINDOW_SECS = 2.0

# Number of top songs to cache in memory
HOT_SONGS_K = 500

# --- CQT-based fingerprinting parameters ---
# Each tuple: (bins_per_octave, hop_length, version_label)
CQT_CONFIGS = [
    (36, 512,  "cqt_freq"),  # high frequency resolution
    (24, 256,  "cqt_time"),  # high time resolution
]
CQT_MIN_FREQ = 55.0        # Minimum frequency (Hz) - A1
CQT_MAX_FREQ = 4000.0      # Maximum frequency (Hz) - Around B7
CQT_THRESHOLD = 3.0        # peak threshold multiplier for CQT
CQT_FILTER_COEF = 0.6      # global amplitude filter coefficient for CQT
CQT_FANOUT = 8             # max target pairings per anchor for CQT
CQT_WINDOW_SECS = 5.0       # max pairing time gap in seconds for CQT