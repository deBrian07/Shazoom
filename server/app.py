import os
import csv
import numpy as np
from quart import Quart, request, jsonify, websocket
from quart_cors import cors
from motor.motor_asyncio import AsyncIOMotorClient
from collections import defaultdict, Counter
from database.utils import (
    audio_file_to_samples,
    generate_fingerprints_multiresolution,
    accumulate_votes_for_hash,
    merge_votes
)
import time
import asyncio
from itertools import chain
from io import BytesIO
import json

app = Quart(__name__)
allowed_origins = ["http://localhost:3000", "https://debrian07.github.io"]
app = cors(app, allow_origin=allowed_origins)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
DEV_MODE = False
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

songs_col = db["songs"]
fingerprints_col = db["fingerprints"]

# Ensure the hash index exists.
@app.before_serving
async def ensure_index():
    await fingerprints_col.create_index("hash")

async def find_fingerprint_batch(batch):
    cursor = fingerprints_col.find(
        {"hash": {"$in": batch}},
        {"hash": 1, "song_id": 1, "offset": 1}
    ).batch_size(1000)
    return await cursor.to_list(length=None)

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
    max_recording = 9.0  
    threshold_time = 4.0 
    min_votes = 40      
    
    await websocket.send(json.dumps({"status": "Recording started"}))
    
    while True:
        remaining = max_recording - (time.time() - start_time)
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
        
        if elapsed > threshold_time and (time.time() - last_match_time) >= 1:
            data = audio_buffer.getvalue()
            buf = BytesIO(data)
            try:
                samples, sample_rate = audio_file_to_samples(buf)
            except Exception as e:
                await websocket.send(json.dumps({"error": str(e)}))
                continue
            
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
            
            # ONE BIG DB QUERY
            db_docs = await fingerprints_col.find(
                {"hash": {"$in": hash_list}},
                {"hash": 1, "song_id": 1, "offset": 1}
            ).to_list(None)
            db_group = defaultdict(list)
            for doc in db_docs:
                h = doc["hash"]
                db_group[h].append((doc["song_id"], doc["offset"]))
            
            # tally votes
            bin_width = 0.2  # seconds
            global_votes = defaultdict(int)
            for h, query_offsets in query_hashes.items():
                if h not in db_group:
                    continue
                query_offsets_array = np.array(query_offsets, dtype=np.float64)
                for song_id, db_offset in db_group[h]:
                    votes_for_hash = accumulate_votes_for_hash(query_offsets_array, db_offset, bin_width)
                    merge_votes(global_votes, votes_for_hash, song_id)
            
            if global_votes:
                vote_counts = Counter(global_votes)
                best_match, best_votes = vote_counts.most_common(1)[0]
                if best_votes >= min_votes:
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
                        break
            last_match_time = time.time()
    
    if not match_result:
        await websocket.send(json.dumps({"status": "No match found after recording"}))
    await websocket.send(json.dumps({"status": "Finished"}))
    await websocket.close(1000)

if __name__ == "__main__":
    # uvicorn app:app --host 0.0.0.0 --port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)