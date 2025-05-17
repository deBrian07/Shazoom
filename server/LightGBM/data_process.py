import os, random, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from utils.utils import audio_file_to_samples, generate_fingerprints_multiresolution

# CONFIG
PROCESSED_CSV = "../download/processed.csv"
WINDOW_SECS   = 2.0
STEP_SECS     = 2.0
MAX_PROCS     = 8

# 1) Load your list of files
df_paths = pd.read_csv(PROCESSED_CSV)
paths = [
    os.path.join("..", "download", p)
    for p in df_paths["wav file location"]
]

def process_song(path):
    """Fingerprint every 2s window of one song."""
    song_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as f:
        samples, sr = audio_file_to_samples(f)
    w = int(WINDOW_SECS * sr)
    s = int(STEP_SECS   * sr)
    recs = []
    for start in range(0, len(samples) - w + 1, s):
        chunk = samples[start:start + w]
        for h, t in generate_fingerprints_multiresolution(chunk, sr):
            recs.append({"hash": h, "song_id": song_id, "label": 1})
    return recs

# 2) POSITIVES in parallel with a progress bar
records = []
with ProcessPoolExecutor(max_workers=MAX_PROCS) as exe:
    # executor.map returns results in input order, so we can wrap in tqdm
    for recs in tqdm(exe.map(process_song, paths),
                     total=len(paths),
                     desc="Fingerprinting songs"):
        records.extend(recs)

# 3) NEGATIVES — just one loop, no duplicates
song_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
for song_id in tqdm(song_ids, desc="Sampling negatives"):
    other = random.choice([s for s in song_ids if s != song_id])
    other_path = next(p for p in paths if os.path.splitext(os.path.basename(p))[0] == other)
    with open(other_path, "rb") as f:
        samples, sr = audio_file_to_samples(f)
    w = int(WINDOW_SECS * sr)
    if len(samples) <= w: 
        continue
    start = np.random.randint(0, len(samples) - w)
    chunk = samples[start:start + w]
    for h, t in generate_fingerprints_multiresolution(chunk, sr):
        records.append({"hash": h, "song_id": song_id, "label": 0})

# 4) Build and save your feature CSV
df = pd.DataFrame(records)
df_df = (
    df[df.label == 1]
      .groupby("hash")["song_id"]
      .nunique()
      .reset_index(name="df")
)
df = df.merge(df_df, on="hash", how="left").fillna({"df": 1})
df["delta_f"], df["family"] = zip(*df["hash"].map(lambda h: (
    abs(int(h.split(':')[1]) - int(h.split(':')[0])),
    h.split(':')[-1]
)))
df[["hash","df","delta_f","family","label"]].to_csv("hash_features.csv", index=False)
print(f"Saved {len(df)} rows to hash_features.csv")
