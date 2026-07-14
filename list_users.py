"""
Utility: list and optionally delete users from the database.
Usage: python list_users.py
       python list_users.py --delete <user_id>
"""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from database.memory_manager import MemoryManager

db = MemoryManager()
users = db.list_users()

print("=== Users in DB ===")
for u in users:
    print(f"  ID: {u['_id']}   Name: {u['name']}")

if "--delete" in sys.argv:
    idx = sys.argv.index("--delete")
    uid = sys.argv[idx + 1]
    db.delete_user(uid)
    print(f"\nDeleted user ID: {uid}")

