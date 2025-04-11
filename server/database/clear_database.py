from pymongo import MongoClient

# Connect to your MongoDB instance.
client = MongoClient("mongodb://localhost:27017/")

DEV_MODE = True  # True when testing something (change to False before commiting)
if DEV_MODE:
    db = client["musicDB_dev"]
else:
    db = client["musicDB"]

# Drop the database.
client.drop_database(db)
print("Database dropped successfully.")