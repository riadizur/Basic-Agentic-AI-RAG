# query_runner.py
import os
import sys
import json
import numpy as np
import faiss
import httpx
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
FAISS_INDEX_PATH = "./faiss.index"
CHUNKS_JSON_PATH = "./chunks.json"
VECTORS_NPY_PATH = "./vectors.npy"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
TOP_K = 5

if not os.path.exists(FAISS_INDEX_PATH):
    print(f"[ERROR] FAISS index not found at {FAISS_INDEX_PATH}. Run embedder.py first.")
    sys.exit(1)

# --- Load all ---
print("[INIT] Loading FAISS index & vectors...")
index = faiss.read_index(FAISS_INDEX_PATH)
vectors = np.load(VECTORS_NPY_PATH)
chunks = json.load(open(CHUNKS_JSON_PATH))
model = SentenceTransformer(EMBED_MODEL)

def query_to_context(query):
    q_embedding = model.encode([query]).astype("float32")
    D, I = index.search(q_embedding, TOP_K)

    selected_chunks = [chunks[i]["chunk"] for i in I[0]]
    context_text = "\n\n---\n\n".join(selected_chunks)
    return context_text

async def ask_ollama(query):
    context = query_to_context(query)
    prompt = f"""You are an enterprise-level AI assistant with access to company documentation. 
Use the following retrieved content to answer the user query. 

Retrieved context:
{context}

User question:
{query}
"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        res = response.json()
        return res['message']['content']

# --- Console Loop ---
import asyncio

async def main():
    print("\n💬 Type your query below. Type 'exit' to quit.")
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n[Thinking... 💡]")
        answer = await ask_ollama(query)
        print(f"\n🧠 Response:\n{answer}")

if __name__ == "__main__":
    asyncio.run(main())