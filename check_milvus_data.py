from pymilvus import MilvusClient

# Initialize Milvus Lite with local SQLite file
client = MilvusClient("milvus_lite.db")

# Get all collections (if unsure which one to use)
collections = client.list_collections()
print("Available collections:", collections)

# Choose your collection name
collection_name = "rag_docs"

# Describe collection (to get field names)
info = client.describe_collection(collection_name)
print("Collection info:", info)

# Query all data (expr="" means no filter)
results = client.query(
    collection_name="rag_docs",
    filter="id >= 0",
    output_fields=["id", "vector", "text", "subject"]
)

# Print results
for item in results:
    print(item)

import json
import numpy as np

# Helper to convert float32, int64, etc.
def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Save to JSON
serializable_results = to_serializable(results)
with open("milvus_results.json", "w", encoding="utf-8") as f:
    json.dump(serializable_results, f, indent=2, ensure_ascii=False)