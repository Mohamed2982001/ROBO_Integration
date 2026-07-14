from qdrant_client import QdrantClient

try:
    client = QdrantClient(url="http://localhost:6333")
    print(f"Client type: {type(client)}")
    print(f"Has search: {hasattr(client, 'search')}")
    print(f"Methods: {[m for m in dir(client) if not m.startswith('_')]}")
except Exception as e:
    print(f"Error: {e}")
