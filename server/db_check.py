from pymongo import MongoClient
from utils.constants import MONGO_URI, DEV_MODE

# connect
client = MongoClient(MONGO_URI)
db = client["musicDB_dev" if DEV_MODE else "musicDB"]

# two options:
# 1) exact count
total = db.songs.count_documents({})

# 2) fast estimate (may be slightly out-of-date)
# total = db.songs.estimated_document_count()

print(f"Total songs in collection: {total}")