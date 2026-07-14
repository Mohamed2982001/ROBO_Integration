import httpx
import json

r = httpx.get("http://localhost:6333/collections/ltm_memories")
print("ltm_memories Collection Info:")
print(json.dumps(r.json(), indent=2))

# Scroll points
r_scroll = httpx.post("http://localhost:6333/collections/ltm_memories/points/scroll", json={"limit": 20})
print("\nPoints in ltm_memories:")
points = r_scroll.json().get("result", {}).get("points", [])
for p in points:
    print(f"ID: {p['id']}, Payload: {p['payload']}")

r_identity = httpx.post("http://localhost:6333/collections/identity_embeddings/points/scroll", json={"limit": 20})
print("\nPoints in identity_embeddings:")
points_id = r_identity.json().get("result", {}).get("points", [])
for p in points_id:
    print(f"ID: {p['id']}, Payload: {p['payload']}")
