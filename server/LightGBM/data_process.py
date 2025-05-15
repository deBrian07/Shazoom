import os
import random
import numpy as np
import pandas as pd
from io import BytesIO

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.utils import audio_file_to_samples, generate_fingerprints_multiresolution

# CONFIG
PROCESSED_CSV = "../download/processed.csv"
WINDOW_SECS   = 2.0
STEP_SECS     = 2.0
MAX_PROCS     = 8

# Load the CSV of paths
df_paths = pd.read_csv(PROCESSED_CSV)
paths = df_paths["wav file location"].tolist()
modified_paths = []

for path in paths:
    modified_paths.append(os.path.join('..', 'download', path))

paths = modified_paths

def process_song(path):
    """Load one song, fingerprint every window, return list of records."""
    song_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as f:
        samples, sr = audio_file_to_samples(f)
    w = int(WINDOW_SECS * sr)
    s = int(STEP_SECS   * sr)
    recs = []
    for start in range(0, len(samples) - w + 1, s):
        chunk = samples[start:start + w]
        fps = generate_fingerprints_multiresolution(chunk, sr)
        for h, t in fps:
            recs.append({"hash": h, "song_id": song_id, "label": 1})
    return recs

# 1) POSITIVES in parallel
records = []
with ProcessPoolExecutor(max_workers=MAX_PROCS) as exe:
    futures = {exe.submit(process_song, p): p for p in paths}
    for fut in as_completed(futures):
        recs = fut.result()
        records.extend(recs)

# 2) NEGATIVES (you can similarly parallelize if desired)
song_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
for song_id in song_ids:
    other = random.choice([s for s in song_ids if s != song_id])
    other_path = df_paths[df_paths["wav file location"].str.contains(other)]["wav file location"].iloc[0]
    with open(other_path, "rb") as f:
        samples, sr = audio_file_to_samples(f)
    w = int(WINDOW_SECS * sr)
    if len(samples) <= w: continue
    start = np.random.randint(0, len(samples) - w)
    chunk = samples[start:start + w]
    fps = generate_fingerprints_multiresolution(chunk, sr)
    for h, t in fps:
        records.append({"hash": h, "song_id": song_id, "label": 0})

# 3) NEGATIVES: for each song, pick one random window from a *different* song
song_ids = df_paths["wav file location"].apply(lambda p: os.path.splitext(os.path.basename(p))[0]).tolist()
for song_id in tqdm(song_ids, desc="Negative examples"):
    other = random.choice([s for s in song_ids if s != song_id])
    other_row = df_paths[df_paths["wav file location"].str.contains(other)].iloc[0]
    with open(other_row["wav file location"], "rb") as f:
        samples, sr = audio_file_to_samples(f)
    if len(samples) <= int(WINDOW_SECS * sr):
        continue
    start = np.random.randint(0, len(samples) - int(WINDOW_SECS * sr))
    chunk = samples[start:start + int(WINDOW_SECS * sr)]
    fps = generate_fingerprints_multiresolution(chunk, sr)
    for h, t in fps:
        records.append({"hash": h, "song_id": song_id, "label": 0})

# 4) Build the feature CSV exactly as before
df = pd.DataFrame(records)
# document frequency on positives
df_df = (
    df[df.label == 1]
      .groupby("hash")["song_id"]
      .nunique()
      .reset_index(name="df")
)
df = df.merge(df_df, on="hash", how="left").fillna({"df": 1})
# parse delta_f and family
df["delta_f"], df["family"] = zip(*df["hash"].map(lambda h: (
    abs(int(h.split(':')[1]) - int(h.split(':')[0])),
    h.split(':')[-1]
)))
df[["hash","df","delta_f","family","label"]].to_csv("hash_features.csv", index=False)
print(f"Saved {len(df)} rows to hash_features.csv")
