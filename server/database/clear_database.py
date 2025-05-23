from pymongo import MongoClient

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from utils.constants import DEV_MODE

# Connect to your MongoDB instance.
client = MongoClient("mongodb://localhost:27017/")

if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

# Drop the database.
client.drop_database(db)
print("Database dropped successfully.")