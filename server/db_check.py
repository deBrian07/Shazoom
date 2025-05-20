from pymongo import MongoClient
from bson import ObjectId
from utils.constants import MONGO_URI, DEV_MODE

# 1) connect
client = MongoClient(MONGO_URI)
db = client["musicDB_dev" if DEV_MODE else "musicDB"]
songs = db["songs"]
fps   = db["fingerprints"]

# 2) find the song and grab its _id
song = songs.find_one({ "title": "a boy is a gun*", "artist": "Tyler, the Creator" })
if not song:
    print("Song not found")
    exit(1)

song_id = song["_id"]

# 3) count fingerprints
total_fps = fps.count_documents({ "song_id": song_id })
print(f"Total fingerprints for '{song['title']}':", total_fps)