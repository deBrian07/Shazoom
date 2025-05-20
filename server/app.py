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
    ALLOWED_ORIGINS, BATCH_SIZE, BIN_WIDTH, DEV_MODE, HOT_SONGS_K,
    MAX_RECORDING, MIN_HASHES, MIN_VOTES, MONGO_URI,
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

# ==================== Normalization parameters ====================
TOP_K_NORMALIZE = 5
TOTAL_THRESHOLD_FACTOR = 1.1  # Multiplier for normalized threshold
song_fp_count = {}
avg_fp_length = 1.0
# ===================================================================

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
    while True:
        cursor = hits_col.find({}, {"song_id": 1, "count": 1}).sort("count", -1).limit(HOT_SONGS_K)
        hot_songs = [doc["song_id"] async for doc in cursor]
        for song_id in hot_songs:
            docs = await fingerprints_col.find(
                {"song_id": song_id},
                {"hash": 1, "song_id": 1, "offset": 1}
            ).to_list(length=None)
            for doc in docs:
                db_cache.setdefault(doc["hash"], []).append((doc["song_id"], doc["offset"]))
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

# MongoDB connection
client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=200)
print(f"Configured MongoDB max pool size: {client.max_pool_size}")
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]
hits_col = db["song_hits"]

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

# Ensure the hash index exists and schedule tasks
@app.before_serving
async def ensure_index():
    # schedule index creation
    asyncio.ensure_future(
        fingerprints_col.create_index([
            ("hash", 1),
            ("song_id", 1),
            ("offset", 1)
        ])
    )
    # tune WiredTiger cache
    cache_gb = 45
    await client.admin.command({
        "setParameter": 1,
        "wiredTigerEngineRuntimeConfig": f"cache_size={cache_gb}G"
    })
    # Precompute fingerprint counts for normalization
    counts = await fingerprints_col.aggregate([
        {"$group": {"_id": "$song_id", "count": {"$sum": 1}}}
    ]).to_list(length=None)
    global song_fp_count, avg_fp_length
    song_fp_count = {doc["_id"]: doc["count"] for doc in counts}
    avg_fp_length = sum(song_fp_count.values()) / len(song_fp_count) if song_fp_count else 1.0
    print(f"Computed fingerprint counts: {len(song_fp_count)} songs, avg={avg_fp_length:.2f}")
    # schedule background tasks
    asyncio.create_task(_prewarm_hot_hashes())
    asyncio.create_task(_monitor_ram())
    asyncio.create_task(_refresh_hot_songs())

async def _prewarm_hot_hashes():
    pipeline = [
        {"$group": {"_id": "$hash", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": TO_PREWARM}
    ]
    cursor = fingerprints_col.aggregate(pipeline)
    hot_hashes = [doc["_id"] async for doc in cursor]
    for i in range(0, len(hot_hashes), BATCH_SIZE):
        batch = hot_hashes[i:i+BATCH_SIZE]
        await fingerprints_col.find(
            {"hash": {"$in": batch}}, {"hash": 1}
        ).batch_size(100).to_list(length=1)

@app.websocket('/stream')
async def stream():
    audio_buffer = BytesIO()
    start_time = time.time()
    last_match_time = time.time()
    match_result = None
    query_hashes = defaultdict(list)
    await websocket.send(json.dumps({"status": "Recording started"}))
    while True:
        remaining = MAX_RECORDING - (time.time() - start_time)
        if remaining <= 0: break
        try:
            chunk = await asyncio.wait_for(websocket.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if isinstance(chunk, str) and chunk.lower() == "end": break
        audio_buffer.write(chunk)
        elapsed = time.time() - start_time
        await websocket.send(json.dumps({"status":"Recording","elapsed":elapsed}))
        if elapsed > THRESHOLD_TIME and (time.time() - last_match_time) >= 1:
            processing_start = time.time()
            with timer("audio_to_samples"):
                samples, sample_rate = audio_file_to_samples(BytesIO(audio_buffer.getvalue()))
            window_size = int(SLIDING_WINDOW_SECS * sample_rate)
            if len(samples) > window_size:
                chunk_samples = samples[-window_size:]
                window_start_time = elapsed - SLIDING_WINDOW_SECS
            else:
                chunk_samples = samples
                window_start_time = 0.0
            with timer("generate_chunk_fps"):
                chunk_fps = generate_fingerprints_multiresolution(chunk_samples, sample_rate)
                print(len(chunk_fps))
            for h, t in chunk_fps:
                query_hashes[h].append(t + window_start_time)
            if not query_hashes:
                await websocket.send(json.dumps({"status":"No new fingerprints yet."}))
                last_match_time = time.time()
                continue
            hash_list = list(query_hashes.keys())
            with timer("db_find"):
                to_fetch = [h for h in hash_list if h not in db_cache]
                batches = [to_fetch[i:i+BATCH_SIZE] for i in range(0,len(to_fetch),BATCH_SIZE)]
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(_process_pool,_fetch_batch_sync,b) for b in batches]
                for fut in asyncio.as_completed(tasks):
                    for doc in await fut:
                        db_cache.setdefault(doc["hash"],[]).append((doc["song_id"],doc["offset"]))
            with timer("build_db_group"):
                db_group = {h:db_cache.get(h,[]) for h in hash_list}
            with timer("compute_idf"):
                total_songs = await songs_col.count_documents({})
                idf_weights = {h:math.log(total_songs/ (len({sid for sid,_ in db_group[h]})+1)) for h in db_group}
            with timer("vote_tally"):
                global_votes = Counter()
                hashes_seen = 0
                for h, qoffs in query_hashes.items():
                    docs = db_group.get(h)
                    if not docs: continue
                    w = idf_weights.get(h,1.0)
                    votes_map = accumulate_votes_vectorized(np.array(qoffs,float),docs,BIN_WIDTH)
                    for key,cnt in votes_map.items():
                        global_votes[key]+=cnt*w
                    hashes_seen+=1
                    if hashes_seen>=MIN_HASHES and len(global_votes)>=2:
                        t2=global_votes.most_common(2)
                        (k1,v1),(k2,v2)=t2
                        if v1>=MIN_VOTES and v1>=v2*1.5: break
            if global_votes:
                # Normalize top candidates
                top = global_votes.most_common(TOP_K_NORMALIZE)
                normed=[]
                for (sid,delta),v in top:
                    total = song_fp_count.get(sid,1)
                    norm_score = v/total
                    normed.append(((sid,delta),v,norm_score))
                (best_sid, best_delta), best_v, best_norm = max(normed, key=lambda x: x[2])
                # dynamic threshold: require more votes for longer songs
                beta = 0.5
                dyn_threshold = MIN_VOTES * (song_fp_count.get(best_sid, 1) / avg_fp_length) ** beta
                if best_v >= dyn_threshold:
                    song = await songs_col.find_one({"_id": best_sid})
                    if song:
                        await hits_col.update_one({"song_id": best_sid}, {"$inc": {"count": 1}}, upsert=True)
                        match_result = {"song": song.get("title"), "artist": song.get("artist"), "offset": best_delta, "raw_votes": best_v}
                        await websocket.send(json.dumps({"status": "Match found", "result": match_result}))
                        
                        break
            last_match_time=time.time()
    if not match_result:
        await websocket.send(json.dumps({"status":"No match found after recording"}))
    await websocket.send(json.dumps({"status":"Finished"}))
    await websocket.close(1000)

async def _monitor_ram():
    while True:
        if psutil.virtual_memory().used>RAM_THRESHOLD_BYTES:
            db_cache.clear();gc.collect();os.execv(sys.executable,[sys.executable]+sys.argv)
        await asyncio.sleep(60)

@app.route('/suggest_song',methods=['POST'])
async def suggest_song():
    data=await request.get_json()
    title = data.get("title", "").strip()
    artist = data.get("artist", "").strip()
    if not title or not artist: return jsonify({"error":"title and artist required"}),400
    row=[time.strftime("%Y-%m-%d %H:%M:%S"),title,artist]
    csv_path="songs_to_add.csv"
    write_header=not os.path.isfile(csv_path)
    with open(csv_path,"a",newline="") as f:
        w=csv.writer(f)
        if write_header: w.writerow(["timestamp","title","artist"])
        w.writerow(row)
    return jsonify({"status":"queued"}),200

if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_level="info",
        workers=WORKERS
    )