from pymongo import MongoClient

# Connect to your MongoDB instance.
client = MongoClient("mongodb://localhost:27017/")

# Select the database you want to remove, e.g., "musicDB".
db = client["musicDB"]

# Drop the database.
client.drop_database(db)
print("Database dropped successfully.")