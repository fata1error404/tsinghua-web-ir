"""
Bulk load all distinct emoji tags from the Emojis table
into the Tags table (one entry per tag).
"""

from pymongo import MongoClient, errors

# — config —
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "emoji-database"
EMOJI_COLL = "Emojis"
TAGS_COLL = "Tags"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
emojis = db[EMOJI_COLL]
tags = db[TAGS_COLL]

# fetch all unique tags from Emojis
all_tags = [t for t in emojis.distinct("tags") if t.strip()]
print(f"Found {len(all_tags)} distinct tags in '{EMOJI_COLL}'.")

# convert list of strings to JSON format
docs = [{"tag": t} for t in all_tags]

# bulk load
try:
    result = tags.insert_many(docs, ordered=False)
    print(f"Inserted {len(result.inserted_ids)} new tags into '{TAGS_COLL}'.")
except errors.BulkWriteError as bwe:
    write_result = bwe.details.get("writeResult", {})
    inserted = write_result.get("nInserted", 0)
    print(
        f"Inserted {inserted} new tags into '{TAGS_COLL}'; {len(docs) - inserted} duplicates skipped."
    )
