import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["robot_memory"]

print("Collections in database:", db.list_collection_names())

for col_name in db.list_collection_names():
    col = db[col_name]
    count = col.count_documents({})
    print(f"\nCollection '{col_name}' has {count} document(s).")
    
    # Print first 10 documents
    print(f"Sample documents from '{col_name}':")
    for doc in col.find().limit(10):
        # Convert ObjectId to string for printing
        doc['_id'] = str(doc['_id'])
        print("  -", doc)
