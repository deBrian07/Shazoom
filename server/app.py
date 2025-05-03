import os
import csv
import numpy as np
import math
from quart import Quart, request, jsonify, websocket
from quart_cors import cors
from motor.motor_asyncio import AsyncIOMotorClient
from collections import defaultdict, Counter
from utils.utils import (
    accumulate_votes_vectorized,
    audio_file_to_samples,
    generate_fingerprints_multiresolution,
    accumulate_votes_for_hash,
    merge_votes
)

from utils.constants import (
    ALLOWED_ORIGINS, BATCH_SIZE, BIN_WIDTH, DEV_MODE,
    MAX_RECORDING, MIN_VOTES, MONGO_URI,
    RAM_THRESHOLD_BYTES, SLIDING_WINDOW_SECS, THRESHOLD_TIME, TO_PREWARM, WORKERS
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
import pymongo
from concurrent.futures import ProcessPoolExecutor

# Shared globals for worker processes
_worker_client = None
_worker_db = None
_worker_col = None

def _init_worker(uri, dev_mode):
    """
    Initializer for process pool: create a MongoClient once per worker process.
    """
    global _worker_client, _worker_db, _worker_col
    _worker_client = pymongo.MongoClient(uri, maxPoolSize=500)
    _worker_db = _worker_client["musicDB_dev" if dev_mode else "musicDB"]
    _worker_col = _worker_db["fingerprints"]

# Synchronous batch fetch using pre-initialized worker client
def _fetch_batch_sync(batch):
    # Use the global _worker_col initialized in each worker
    docs = list(_worker_col.find(
        {"hash": {"$in": batch}},
        {"hash": 1, "song_id": 1, "offset": 1}
    ))
    return docs

# Process pool executor for DB fetch parallelism with initializer
_process_pool = ProcessPoolExecutor(
    max_workers=16,
    initializer=_init_worker,
    initargs=(MONGO_URI, DEV_MODE)
)

import sys
import gc

# --- Background: refresh hot songs cache ---
async def _refresh_hot_songs():
    """
    Periodically load the top HOT_SONGS_K frequent songs and cache their fingerprints.
    """
    while True:
        # fetch top-K songs by persistent count
        cursor = hits_col.find({}, {"song_id": 1, "count": 1}).sort("count", -1).limit(HOT_SONGS_K)
        hot_songs = [doc["song_id"] async for doc in cursor]
        # preload fingerprints for each hot song
        for song_id in hot_songs:
            docs = await fingerprints_col.find(
                {"song_id": song_id},
                {"hash": 1, "song_id": 1, "offset": 1}
            ).to_list(length=None)
            for doc in docs:
                db_cache.setdefault(doc["hash"], []).append((doc["song_id"], doc["offset"]))
        # wait an 5 mins before refreshing
        await asyncio.sleep(300)

from contextlib import contextmanager



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
client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=200)  # increased pool size to 200
# inspect MongoDB connection pool configuration
print(f"Configured MongoDB max pool size: {client.max_pool_size}")
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]
# Persistent collection for tracking match counts
hits_col = db["song_hits"]
# Number of top songs to cache in memory
HOT_SONGS_K = 1000

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
    # tune WiredTiger cache to a fixed 20GB per worker
    cache_gb = 45  # fixed cache size per worker in GB
    await client.admin.command({
        "setParameter": 1,
        "wiredTigerEngineRuntimeConfig": f"cache_size={cache_gb}G"
    })
    print(f"WiredTiger cache size set to {cache_gb}G")
    # schedule cache prewarm, RAM monitor, and hot-songs refresh tasks
    asyncio.create_task(_prewarm_hot_hashes())
    asyncio.create_task(_monitor_ram())
    asyncio.create_task(_refresh_hot_songs())

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
      - Record for up to MAX_RECORDING seconds.
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
            # ---- fingerprint generation (sliding window incremental) ----
            # mark processing start time
            processing_start = time.time()
            # compute samples and rate from full buffer
            with timer("audio_to_samples"):
                samples, sample_rate = audio_file_to_samples(buf)
            # extract only the last SLIDING_WINDOW_SECS of audio
            window_size = int(SLIDING_WINDOW_SECS * sample_rate)
            if len(samples) > window_size:
                chunk_samples = samples[-window_size:]
                window_start_time = elapsed - SLIDING_WINDOW_SECS
            else:
                chunk_samples = samples
                window_start_time = 0.0
            # generate fingerprints on the recent chunk
            with timer("generate_chunk_fps"):
                chunk_fps = generate_fingerprints_multiresolution(chunk_samples, sample_rate)
            # adjust offsets to global timeline
            query_fps = [(h, t + window_start_time) for (h, t) in chunk_fps]
            if not query_fps:
                await websocket.send(json.dumps({"status": "No new fingerprints yet."}))
                last_match_time = time.time()
                continue
            
            # group query hashes
            query_hashes = defaultdict(list)
            for h, q_offset in query_fps:
                query_hashes[h].append(q_offset)
            hash_list = list(query_hashes.keys())

            # ---- cached DB lookup for new hashes (process-parallel) ----
            with timer("db_find"):
                to_fetch = [h for h in hash_list if h not in db_cache]
                # split into batches to avoid huge $in lists
                batches = [to_fetch[i:i+BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE) if to_fetch[i:i+BATCH_SIZE]]
                # use process pool to fetch batches concurrently
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(_process_pool, _fetch_batch_sync, batch)
                    for batch in batches
                ]
                for future in asyncio.as_completed(tasks):
                    fresh_docs = await future
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

            # ---- vote tallying (vectorized with early exit) ----
            with timer("vote_tally"):
                global_votes = Counter()
                early = False
                early_match = None
                for h, query_offsets in query_hashes.items():
                    docs = db_group.get(h)
                    if not docs:
                        continue
                    w = idf_weights.get(h, 1.0)
                    q_arr = np.array(query_offsets, dtype=np.float64)
                    votes = accumulate_votes_vectorized(q_arr, docs, BIN_WIDTH)
                    for (song_id, delta), cnt in votes.items():
                        key = (song_id, delta)
                        global_votes[key] += cnt * w
                        if global_votes[key] >= MIN_VOTES:
                            early = True
                            early_match = key
                            break
                    if early:
                        break

            if global_votes:
                best_match, best_votes = global_votes.most_common(1)[0]
                if best_votes >= MIN_VOTES:
                    best_song_id, best_delta = best_match
                    song = await songs_col.find_one({"_id": best_song_id})
                    if song:
                        # increment persistent hit count
                        await hits_col.update_one(
                            {"song_id": best_song_id},
                            {"$inc": {"count": 1}},
                            upsert=True
                        )
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
      
if __name__ == "__main__":
    # In production, use an ASGI server with multiple workers for concurrency
    import uvicorn
    uvicorn.run(
        "app:app",  # use import string so workers are enabled
        host="0.0.0.0",
        port=5000,
        log_level="info",
        workers=WORKERS  # use physical core count  # adjust this based on available CPU cores
    )
