import os
import csv
import numpy as np
import math
from quart import Quart, request, jsonify, websocket
from quart_cors import cors
from motor.motor_asyncio import AsyncIOMotorClient
from collections import defaultdict, Counter
from utils.utils import (
    audio_file_to_samples,
    generate_fingerprints_multiresolution,
    accumulate_votes_for_hash,
    merge_votes
)
import time
import asyncio
import uvloop
# Use uvloop for a faster asyncio event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from itertools import chain
from io import BytesIO
import json
import psutil
import sys
import gc
from contextlib import contextmanager

from utils.constants import (
    ALLOWED_ORIGINS, BATCH_SIZE, BIN_WIDTH, DEV_MODE,
    MAX_RECORDING, MIN_VOTES, MONGO_URI,
    RAM_THRESHOLD_BYTES, THRESHOLD_TIME, TO_PREWARM
)

# --- Timing helper ---
@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] {t1 - t0:.3f}s")

app = Quart(__name__)
app = cors(app, allow_origin=ALLOWED_ORIGINS)

# MongoDB connection
client = AsyncIOMotorClient(MONGO_URI)
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]

db_cache = {}

async def find_fingerprint_batch(batch):
    key = tuple(batch)
    if key in db_cache:
        return db_cache[key]
    cursor = fingerprints_col.find(
        {"hash": {"$in": batch}},
        {"hash": 1, "song_id": 1, "offset": 1}
    ).batch_size(1000)
    res = await cursor.to_list(length=None)
    db_cache[key] = res
    return res

# Ensure the hash index exists and schedule RAM monitor and cache prewarm
@app.before_serving
async def ensure_index():
    # schedule covered compound index creation in the background
    asyncio.ensure_future(
        fingerprints_col.create_index([
            ("hash", 1),
            ("song_id", 1),
            ("offset", 1)
        ])
    )
    # schedule cache prewarm and RAM monitor tasks
    asyncio.create_task(_prewarm_hot_hashes())
    asyncio.create_task(_monitor_ram())

async def _prewarm_hot_hashes():
    # identify most frequent hashes (for OS cache warm-up only)
    pipeline = [
        {"$group": {"_id": "$hash", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": TO_PREWARM}
    ]
    cursor = fingerprints_col.aggregate(pipeline)
    hot_hashes = [doc["_id"] async for doc in cursor]

    # warm index and data in OS cache in small projected batches
    for i in range(0, len(hot_hashes), BATCH_SIZE):
        batch = hot_hashes[i:i+BATCH_SIZE]
        # fetch keys only (no offsets/song_ids) to touch pages
        await fingerprints_col.find(
            {"hash": {"$in": batch}},
            {"hash": 1}
        ).batch_size(100).to_list(length=1)

@app.websocket('/stream')
async def stream():
    """
    WebSocket endpoint for real-time streaming recognition.
    Logic:
      - Start receiving audio chunks from the client.
      - Record for up to 15 seconds (max_recording).
      - Do not try matching until at least 5 seconds have elapsed.
      - Every second thereafter, process the currently accumulated audio,
        generate fingerprints, query the DB, and compute votes.
      - If the raw vote count is at least 40, immediately send the match result.
      - If no valid match is found by the end, send "No match found."
    """
    audio_buffer = BytesIO()
    start_time = time.time()
    last_match_time = time.time()
    match_result = None
    
    await websocket.send(json.dumps({"status": "Recording started"}))
    
    while True:
        remaining = MAX_RECORDING - (time.time() - start_time)
        if remaining <= 0:
            break
        try:
            chunk = await asyncio.wait_for(websocket.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            break  
        if isinstance(chunk, str) and chunk.lower() == "end":
            break
        audio_buffer.write(chunk)
        elapsed = time.time() - start_time
        await websocket.send(json.dumps({"status": "Recording", "elapsed": elapsed}))
        
        if elapsed > THRESHOLD_TIME and (time.time() - last_match_time) >= 1:
            data = audio_buffer.getvalue()
            buf = BytesIO(data)
            try:
                samples, sample_rate = audio_file_to_samples(buf)
            except Exception as e:
                await websocket.send(json.dumps({"error": str(e)}))
                continue
            
            # track processing start
            processing_start = time.time()

            # ---- fingerprint generation ----
            with timer("generate_fps"):
                query_fps = generate_fingerprints_multiresolution(samples, sample_rate)
            if not query_fps:
                await websocket.send(json.dumps({"status": "No fingerprints yet."}))
                last_match_time = time.time()
                continue
            
            # group query hashes
            query_hashes = defaultdict(list)
            for h, q_offset in query_fps:
                query_hashes[h].append(q_offset)
            hash_list = list(query_hashes.keys())

            # ---- cached DB lookup for new hashes ----
            with timer("db_find"):
                to_fetch = [h for h in hash_list if h not in db_cache]
                # split into batches to avoid huge $in lists
                for i in range(0, len(to_fetch), BATCH_SIZE):
                    batch = to_fetch[i:i+BATCH_SIZE]
                    if not batch:
                        continue
                    cursor = fingerprints_col.find(
                        {"hash": {"$in": batch}},
                        {"hash": 1, "song_id": 1, "offset": 1}
                    )
                    fresh_docs = await cursor.to_list(length=None)
                    for doc in fresh_docs:
                        db_cache.setdefault(doc["hash"], []).append((doc["song_id"], doc["offset"]))

            # ---- in-memory grouping ----
            with timer("build_db_group"):
                db_group = {h: db_cache.get(h, []) for h in hash_list}
            
            # ---- IDF weight computation ----
            with timer("compute_idf"):
                total_songs = await songs_col.count_documents({})
                idf_weights = {
                    h: math.log(total_songs / (len({sid for sid, _ in db_group[h]}) + 1))
                    for h in db_group
                }

            # ---- vote tallying ----
            with timer("vote_tally"):
                global_votes = Counter()
                for h, query_offsets in query_hashes.items():
                    if h not in db_group:
                        continue
                    w = idf_weights.get(h, 1.0)
                    q_arr = np.array(query_offsets, dtype=np.float64)
                    for song_id, db_offset in db_group[h]:
                        votes_for_hash = accumulate_votes_for_hash(q_arr, db_offset, BIN_WIDTH)
                        for delta, cnt in votes_for_hash.items():
                            global_votes[(song_id, delta)] += cnt * w

            if global_votes:
                best_match, best_votes = global_votes.most_common(1)[0]
                if best_votes >= MIN_VOTES:
                    best_song_id, best_delta = best_match
                    song = await songs_col.find_one({"_id": best_song_id})
                    if song:
                        match_result = {
                            "song": song.get("title"),
                            "artist": song.get("artist"),
                            "offset": best_delta,
                            "raw_votes": best_votes
                        }
                        await websocket.send(json.dumps({"status": "Match found", "result": match_result}))
                        processing_end = time.time()
                        print(f"Processing time: {processing_end - processing_start:.2f} seconds")
                        break
            last_match_time = time.time()
    
    if not match_result:
        await websocket.send(json.dumps({"status": "No match found after recording"}))
    await websocket.send(json.dumps({"status": "Finished"}))
    await websocket.close(1000)

async def _monitor_ram():
    # Clear cache if RAM usage exceeds threshold
    while True:
        mem = psutil.virtual_memory()
        if mem.used > RAM_THRESHOLD_BYTES:
            print(f"RAM usage {mem.used/(1024**3):.1f}GB > threshold, clearing cache")
            db_cache.clear()
            gc.collect()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        await asyncio.sleep(60)

# (Removed duplicate _monitor_ram)():
    while True:
        if psutil.virtual_memory().used > RAM_THRESHOLD_BYTES:
            print(f"RAM usage {psutil.virtual_memory().used/(1024**3):.1f}GB > 58GB, restarting")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        await asyncio.sleep(60)

if __name__ == "__main__":
    # For local development only; in production use an ASGI server
    app.run(host="0.0.0.0", port=5000, debug=False)
