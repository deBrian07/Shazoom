# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017"
ALLOWED_ORIGINS = ["http://localhost:3000", "https://debrian07.github.io"]


DEV_MODE = True  # True when testing something (change to False before commiting)

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
MIN_VOTES = 30             # min vote count to accept a match
BIN_WIDTH = 0.2            # time-delta quantization width (seconds)
MAX_RECORDING = 9.0        # max recording duration (seconds)
THRESHOLD_TIME = 4.0       # initial wait time before matching (seconds)
MIN_HASHES = 10

# System parameters
RAM_THRESHOLD_BYTES = 58 * 1024 ** 3  # 58 GB, used in RAM-monitoring

TO_PREWARM = 5000  # reduce if memory is tight
BATCH_SIZE = 1000

WORKERS = 8


# define sliding window length in seconds
SLIDING_WINDOW_SECS = 2.0